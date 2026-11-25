"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

REFUSAL_TEXT = "В базе знаний не найдено."
MAX_FACTS_CHARS = 8000

LOCAL_SYSTEM = (
    "Ты ассистент сервисной службы «ПолярТех». Отвечай на вопрос строго по пронумерованным фактам ниже. "
    "Не выдумывай коды и детали. Если в фактах есть цепочка замен REPLACED_BY, назови актуальную (последнюю) деталь. "
    "Если ответа в фактах нет, напиши: «В базе знаний не найдено.»"
)

MAP_SYSTEM = (
    "Ты анализируешь сводку группы документов «ПолярТех». "
    "Оцени, насколько сводка помогает ответить на вопрос (score от 0 до 10, где 0 — совсем не относится к вопросу), "
    "и сформулируй краткий частичный ответ строго по фактам из сводки."
)

REDUCE_SYSTEM = (
    "Ты ассистент сервисной службы «ПолярТех». Объедини частичные ответы в один точный и связный итоговый ответ. "
    "Опирайся только на переданные данные, не выдумывай факты и числа."
)


def _edge_fact(graph: nx.DiGraph, u: str, v: str) -> str:
    data = graph.edges[u, v]
    u_name = graph.nodes[u]["name"]
    v_name = graph.nodes[v]["name"]
    fact = f"{u_name} - {data['type']}-> {v_name}"
    if data["type"] == "FAILED" and data.get("weight", 1) > 1:
        fact += f" (отказов: {data['weight']})"
    return fact


def _collect_replacements(graph: nx.DiGraph, start_nodes: set[str]) -> list[tuple[str, str]]:
    """Собирает все исходящие цепочки REPLACED_BY от узлов типа Part."""
    edges: list[tuple[str, str]] = []
    queue = [n for n in start_nodes if graph.nodes[n].get("type") == "Part"]
    visited = set(queue)
    while queue:
        curr = queue.pop(0)
        for nxt in graph.successors(curr):
            if graph.edges[curr, nxt].get("type") == "REPLACED_BY":
                edges.append((curr, nxt))
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
    return edges


def _extract_local_edges(graph: nx.DiGraph, anchors: list[str]) -> list[tuple[str, str]]:
    undirected = graph.to_undirected(as_view=True)
    selected: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add_edge(u: str, v: str) -> None:
        if (u, v) not in seen:
            seen.add((u, v))
            selected.append((u, v))

    if len(anchors) >= 2:
        path_nodes: set[str] = set(anchors)
        for src, dst in combinations(anchors, 2):
            for path in nx.all_simple_paths(undirected, src, dst, cutoff=3):
                path_nodes.update(path)
                for a, b in zip(path, path[1:]):
                    if graph.has_edge(a, b):
                        add_edge(a, b)
                    if graph.has_edge(b, a):
                        add_edge(b, a)
        for u, v in _collect_replacements(graph, path_nodes):
            add_edge(u, v)
    else:
        anchor = anchors[0]
        neighborhood = set(nx.single_source_shortest_path_length(undirected, anchor, cutoff=2))
        for u in neighborhood:
            for v in graph.successors(u):
                if v in neighborhood:
                    add_edge(u, v)
        for u, v in _collect_replacements(graph, neighborhood):
            add_edge(u, v)

    return selected


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    graph = kb.graph
    codes = find_codes(question)
    anchors: list[str] = []

    # 1. Проверяем коды из вопроса: любой неизвестный код -> немедленный отказ без вызова LLM
    for code in codes:
        node_id = f"{code_type(code)}:{code}"
        if node_id not in graph:
            return Answer(REFUSAL_TEXT, refused=True)
        if node_id not in anchors:
            anchors.append(node_id)

    # 2. Добавляем словарные названия, присутствующие в графе
    for kind, name in find_names(question, kb.vocabulary):
        node_id = f"{kind}:{name}"
        if node_id in graph and node_id not in anchors:
            anchors.append(node_id)

    if not anchors:
        return Answer(REFUSAL_TEXT, refused=True)

    # 3. Отбираем рёбра (пути между якорями или окрестность)
    edges = _extract_local_edges(graph, anchors)
    if not edges:
        return Answer(REFUSAL_TEXT, refused=True)

    # 4. Формируем пронумерованный список фактов (до 8000 символов) и список документов-источников
    fact_lines: list[str] = []
    sources: list[str] = []
    seen_sources: set[str] = set()
    total_chars = 0

    for idx, (u, v) in enumerate(edges, start=1):
        line = f"{idx}. {_edge_fact(graph, u, v)}"
        if total_chars + len(line) + 1 > MAX_FACTS_CHARS:
            break
        fact_lines.append(line)
        total_chars += len(line) + 1
        for chunk_id in graph.edges[u, v].get("sources", []):
            doc_id = chunk_id.split("#", 1)[0]
            if doc_id not in seen_sources:
                seen_sources.add(doc_id)
                sources.append(doc_id)

    if not fact_lines:
        return Answer(REFUSAL_TEXT, refused=True)

    facts_block = "\n".join(fact_lines)
    reply = llm.chat(
        [
            {"role": "system", "content": LOCAL_SYSTEM},
            {"role": "user", "content": f"Факты из базы знаний:\n{facts_block}\n\nВопрос: {question}"},
        ]
    )
    text = reply.content.strip()
    refused = "не найдено" in text.lower()
    return Answer(text=text, sources=[] if refused else sources, facts=fact_lines, refused=refused)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    
    useful: list[tuple[int, str, list[str]]] = []

    # Шаг Map: опрашиваем каждое сообщество с непустой сводкой
    for community in kb.communities:
        if not community.summary.strip():
            continue
        reply = llm.chat(
            [
                {"role": "system", "content": MAP_SYSTEM},
                {"role": "user", "content": f"Сводка:\n{community.summary}\n\nВопрос: {question}"},
            ],
            json_schema=MAP_SCHEMA,
        )
        try:
            data = json.loads(reply.content)
            score = int(data.get("score", 0))
            partial = str(data.get("answer", "")).strip()
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

        if score > 0 and partial:
            comm_docs: list[str] = []
            for node_id in community.nodes:
                if node_id in kb.graph:
                    for chunk_id in kb.graph.nodes[node_id].get("sources", []):
                        doc_id = chunk_id.split("#", 1)[0]
                        if doc_id not in comm_docs:
                            comm_docs.append(doc_id)
            useful.append((score, partial, comm_docs))

    if not useful:
        return Answer(REFUSAL_TEXT, refused=True)

    # Сортируем по убыванию оценки полезности
    useful.sort(key=lambda item: item[0], reverse=True)

    partial_block = "\n\n".join(f"[Оценка {score}/10]: {ans}" for score, ans, _ in useful)
    sources: list[str] = []
    seen_sources: set[str] = set()
    for _, _, docs in useful:
        for doc_id in docs:
            if doc_id not in seen_sources:
                seen_sources.add(doc_id)
                sources.append(doc_id)

    # Шаг Reduce: объединяем полезные частичные ответы
    final_reply = llm.chat(
        [
            {"role": "system", "content": REDUCE_SYSTEM},
            {"role": "user", "content": f"Частичные ответы:\n{partial_block}\n\nВопрос: {question}"},
        ]
    )
    text = final_reply.content.strip()
    refused = "не найдено" in text.lower()
    return Answer(text=text, sources=[] if refused else sources, refused=refused)
