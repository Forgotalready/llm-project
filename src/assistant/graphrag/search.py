"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

MAX_FACT_CHARS = 8000
MAX_FACTS = 60
MAX_PATH_LEN = 3
MAX_PATHS_PER_PAIR = 12
NEIGHBOUR_DEPTH = 2
MIN_SCORE = 5
MAX_PARTIALS = 5
REFUSAL = "В базе знаний не найдено."

FACT_PRIORITY = {
    "REPLACED_BY": 0,
    "BELONGS_TO": 1,
    "HAS_PART": 1,
    "IS_A": 1,
    "INDICATES": 1,
    "FIXED_BY": 1,
    "SUPPLIED_BY": 2,
    "FAILED": 3,
}

EMPTY_MARKERS = (
    "не найдено",
    "не указано",
    "не указана",
    "не указаны",
    "нет данных",
    "нет информации",
    "отсутствует",
    "не упоминается",
    "не содержится",
    "невозможно",
    "не может",
)

LOCAL_SYSTEM = (
    "Ты ассистент сервисной службы «ПолярТех». Ответь ОДНИМ коротким предложением по фактам ниже. "
    "Не рассуждай вслух, не пересказывай список и не пиши номера фактов. "
    "Коды пиши точно как в фактах: регламенты начинаются с SP, ошибки с E, детали с P, модели с AR, BZ, GS. "
    "Поставщик и линейка — это название, а не код. "
    "Если в фактах есть цепочка REPLACED_BY, назови последнюю деталь цепочки. "
    "Если нужного факта в списке нет, ответь ровно: «в базе знаний не найдено»."
)

MAP_SYSTEM = (
    "Ты ассистент сервисной службы «ПолярТех». По сводке сообщества ответь на вопрос. "
    'Верни только JSON вида {"answer": "...", "score": N}. '
    "Если сводка содержит ответ: score от 7 до 10, answer — одно короткое предложение "
    "с фактами и числами из сводки. "
    'Если сводка не относится к вопросу или ответа в ней нет: {"answer": "", "score": 0}. '
    "Не пиши в answer слова «не указано», «не найдено», «нет данных» — вместо этого ставь score 0. "
    "Не пиши рассуждения и текст вне JSON."
)

REDUCE_SYSTEM = (
    "Ты ассистент сервисной службы «ПолярТех». Из частичных ответов собери итоговый ответ на вопрос. "
    "Отвечай ОДНИМ коротким предложением, без рассуждений и без пересказа списка. "
    "Коды и числа пиши точно как в частичных ответах, ничего не выдумывай. "
    "Отсутствие данных в одной сводке не противоречие: просто не учитывай её. "
    "Если ответы расходятся, выбери тот, где больше данных и выше полезность. "
    "Если полезных ответов несколько, объедини их факты в одно предложение. "
    "Только если данных нет ни в одном ответе, напиши: «в базе знаний не найдено»."
)


def parse_json_object(raw: str) -> dict | None:
    """Возвращает dict из JSON-ответа модели или None, если ответ не разбирается."""
    text = raw.strip()
    if text.startswith("```"):
        text = text[3:]
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.split("```", 1)[0]
    try:
        data = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _select_facts(question: str, kb: KnowledgeBase) -> list[tuple[str, str, str]] | None:
    """Отбирает рёбра графа, которые имеют право попасть в контекст модели."""
    graph = kb.graph
    codes = find_codes(question)
    for code in codes:
        if f"{code_type(code)}:{code}" not in graph:
            return None
    anchors = [f"{code_type(code)}:{code}" for code in codes]
    anchors += [
        f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary) if f"{kind}:{name}" in graph
    ]
    anchors = list(dict.fromkeys(anchors))
    if not anchors:
        return None

    undirected = graph.to_undirected()
    facts: dict[tuple[str, str, str], None] = {}

    def add(u: str, v: str) -> None:
        if graph.has_edge(u, v):
            facts.setdefault((u, graph.edges[u, v]["type"], v), None)
        else:
            facts.setdefault((v, graph.edges[v, u]["type"], u), None)

    if len(anchors) == 1:
        frontier = {anchors[0]}
        visited = set(frontier)
        for _ in range(NEIGHBOUR_DEPTH):
            next_frontier: set[str] = set()
            for node in frontier:
                for neighbour in undirected.neighbors(node):
                    add(node, neighbour)
                    if neighbour not in visited:
                        next_frontier.add(neighbour)
            visited |= next_frontier
            frontier = next_frontier
    else:
        path_nodes: set[str] = set()
        for a, b in combinations(anchors, 2):
            for index, path in enumerate(nx.all_simple_paths(undirected, a, b, cutoff=MAX_PATH_LEN)):
                if index >= MAX_PATHS_PER_PAIR:
                    break
                path_nodes.update(path)
                for u, v in zip(path, path[1:], strict=False):
                    add(u, v)
        for node in sorted(path_nodes):
            if graph.nodes[node]["type"] != "Part":
                continue
            current, seen = node, {node}
            while True:
                successors = [
                    v
                    for v in graph.successors(current)
                    if graph.edges[current, v]["type"] == "REPLACED_BY" and v not in seen
                ]
                if not successors:
                    break
                facts.setdefault((current, "REPLACED_BY", successors[0]), None)
                seen.add(successors[0])
                current = successors[0]
            for target in graph.successors(current):
                kind = graph.edges[current, target]["type"]
                if kind in {"IS_A", "SUPPLIED_BY"}:
                    facts.setdefault((current, kind, target), None)

    priority = dict(FACT_PRIORITY)
    if any(graph.nodes[anchor]["type"] == "ProductLine" for anchor in anchors):
        priority.update({"FAILED": 0, "HAS_PART": 4, "IS_A": 4, "SUPPLIED_BY": 4})
    ordered = sorted(facts, key=lambda fact: (priority.get(fact[1], 5), fact[0], fact[2]))
    return ordered[:MAX_FACTS]


def _fact_line(graph: nx.DiGraph, u: str, kind: str, v: str) -> str:
    line = f"{graph.nodes[u]['name']} -{kind}-> {graph.nodes[v]['name']}"
    if kind == "FAILED":
        line += f" (отчётов: {graph.edges[u, v].get('weight', 1)})"
    return line


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа."""
    facts = _select_facts(question, kb)
    if not facts:
        return Answer(REFUSAL, refused=True)

    lines: list[str] = []
    sources: list[str] = []
    budget = MAX_FACT_CHARS
    for u, kind, v in facts:
        line = _fact_line(kb.graph, u, kind, v)
        if len(line) + 1 > budget:
            break
        budget -= len(line) + 1
        lines.append(line)
        for chunk in kb.graph.edges[u, v].get("sources", []):
            doc = chunk.split("#")[0]
            if doc not in sources:
                sources.append(doc)
    if not lines:
        return Answer(REFUSAL, refused=True)

    numbered = "\n".join(f"{i}. {line}" for i, line in enumerate(lines, start=1))
    reply = llm.chat(
        [
            {"role": "system", "content": LOCAL_SYSTEM},
            {"role": "user", "content": f"Факты:\n{numbered}\n\nВопрос: {question}"},
        ]
    )
    text = reply.content.strip()
    return Answer(text, sources=sources, facts=lines, refused="не найдено" in text.lower())


def _parse_partial(raw: str) -> tuple[str, int] | None:
    data = parse_json_object(raw)
    if data is None:
        return None
    answer, score = data.get("answer"), data.get("score")
    if not isinstance(answer, str) or isinstance(score, bool) or not isinstance(score, (int, float)):
        return None
    return answer.strip(), max(0, min(10, int(score)))


def _useful(answer: str, score: int) -> bool:
    if score < MIN_SCORE or not answer:
        return False
    lowered = answer.lower()
    return not any(marker in lowered for marker in EMPTY_MARKERS)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага (map/reduce)."""
    partials: list[tuple[int, str]] = []
    sources: list[str] = []
    for community in kb.communities:
        sub_graph = kb.graph.subgraph(community.nodes)
        if not community.summary or sub_graph.number_of_edges() == 0:
            continue
        reply = llm.chat(
            [
                {"role": "system", "content": MAP_SYSTEM},
                {"role": "user", "content": f"Сводка сообщества:\n{community.summary}\n\nВопрос: {question}"},
            ],
            json_schema=MAP_SCHEMA,
        )
        parsed = _parse_partial(reply.content)
        if parsed is None:
            continue
        answer, score = parsed
        if not _useful(answer, score):
            continue
        partials.append((score, answer))
        for u, v in sub_graph.edges:
            for chunk in sub_graph.edges[u, v].get("sources", []):
                doc = chunk.split("#")[0]
                if doc not in sources:
                    sources.append(doc)
    if not partials:
        return Answer(REFUSAL, refused=True)

    partials.sort(key=lambda item: -item[0])
    partials = partials[:MAX_PARTIALS]
    numbered = "\n".join(f"{i}. (полезность {score}) {answer}" for i, (score, answer) in enumerate(partials, start=1))
    reply = llm.chat(
        [
            {"role": "system", "content": REDUCE_SYSTEM},
            {"role": "user", "content": f"Частичные ответы:\n{numbered}\n\nВопрос: {question}"},
        ]
    )
    text = reply.content.strip()
    return Answer(text, sources=sources, refused="не найдено" in text.lower())