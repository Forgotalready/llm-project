"""Сообщества графа и их сводки. Требование FR-3."""

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

SEED = 7
MAX_WORDS = 150
MAX_RELATIONS = 60

SYSTEM = (
    "Напиши сводку сообщества не длиннее 150 слов. "
    "Первым предложением назови линейку и модели, которые в него входят. "
    "Вторым — узел с наибольшим числом отказов и это число. "
    "Третьим — модель с наибольшим числом отказов и это число, если такая модель есть. "
    "Дальше назови детали, снятые с производства, и линейки, которым они принадлежат. "
    "Потом состав и связи. Только приведённые факты и числа, по-русски."
)


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным seed."""
    if graph.number_of_nodes() == 0:
        return []
    groups = nx.community.louvain_communities(graph.to_undirected(), weight="weight", seed=SEED)
    ordered = sorted((sorted(group) for group in groups), key=lambda nodes: nodes)
    return [Community(community_id, nodes) for community_id, nodes in enumerate(ordered)]


def _entity_lines(sub_graph: nx.DiGraph) -> tuple[str, dict[str, list[str]]]:
    types: dict[str, list[str]] = {}
    for node in sorted(sub_graph.nodes):
        types.setdefault(sub_graph.nodes[node]["type"], []).append(sub_graph.nodes[node]["name"])
    lines = "\n".join(f"- {kind}: {', '.join(names)}" for kind, names in sorted(types.items()))
    return lines, types


def _relation_lines(sub_graph: nx.DiGraph) -> str:
    edges = sorted(sub_graph.edges(data=True), key=lambda item: (-item[2].get("weight", 1), item[0], item[1]))
    return "\n".join(
        f"- {sub_graph.nodes[u]['name']} -{data['type']}-> {sub_graph.nodes[v]['name']} "
        f"(вес {data.get('weight', 1)})"
        for u, v, data in edges[:MAX_RELATIONS]
    )


def _component_totals(sub_graph: nx.DiGraph) -> str:
    totals: dict[str, int] = {}
    for _, target, data in sub_graph.edges(data=True):
        if data["type"] == "FAILED":
            name = sub_graph.nodes[target]["name"]
            totals[name] = totals.get(name, 0) + data.get("weight", 1)
    return "\n".join(
        f"- {name}: {count}" for name, count in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    )


def _model_totals(sub_graph: nx.DiGraph) -> str:
    totals: dict[str, int] = {}
    for source, _, data in sub_graph.edges(data=True):
        if data["type"] == "FAILED":
            name = sub_graph.nodes[source]["name"]
            totals[name] = totals.get(name, 0) + data.get("weight", 1)
    return "\n".join(
        f"- {name}: {count}" for name, count in sorted(totals.items(), key=lambda item: (-item[1], item[0]))
    )


def _replacement_lines(sub_graph: nx.DiGraph) -> str:
    chains: list[str] = []
    for node in sorted(sub_graph.nodes):
        if sub_graph.nodes[node]["type"] != "Part":
            continue
        if not any(sub_graph.edges[node, v]["type"] == "REPLACED_BY" for v in sub_graph.successors(node)):
            continue
        if any(sub_graph.edges[u, node]["type"] == "REPLACED_BY" for u in sub_graph.predecessors(node)):
            continue
        chain = [sub_graph.nodes[node]["name"]]
        current, seen = node, {node}
        while True:
            successors = [
                v
                for v in sub_graph.successors(current)
                if sub_graph.edges[current, v]["type"] == "REPLACED_BY" and v not in seen
            ]
            if not successors:
                break
            current = successors[0]
            seen.add(current)
            chain.append(sub_graph.nodes[current]["name"])
        chains.append(f"- {' -> '.join(chain)}: первая деталь снята с производства, последняя актуальна")
    return "\n".join(chains)


def _line_parts(community: Community, graph: nx.DiGraph) -> str:
    lines: dict[str, list[str]] = {}
    for node in sorted(community.nodes):
        if graph.nodes[node]["type"] != "Part":
            continue
        for model in graph.predecessors(node):
            if graph.edges[model, node]["type"] != "HAS_PART":
                continue
            for line in graph.successors(model):
                if graph.edges[model, line]["type"] != "BELONGS_TO":
                    continue
                name = graph.nodes[line]["name"]
                part = graph.nodes[node]["name"]
                bucket = lines.setdefault(name, [])
                if part not in bucket:
                    bucket.append(part)
    return "\n".join(f"- {name}: {', '.join(sorted(parts))}" for name, parts in sorted(lines.items()))


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов. Все числа считает код по весам рёбер."""
    sub_graph = graph.subgraph(community.nodes)
    entities, types = _entity_lines(sub_graph)
    relations = _relation_lines(sub_graph)
    components = _component_totals(sub_graph)
    models = _model_totals(sub_graph)
    replacements = _replacement_lines(sub_graph)
    by_line = _line_parts(community, graph)

    prompt = (
        f"Сообщество {community.community_id}: узлов {sub_graph.number_of_nodes()}, "
        f"связей {sub_graph.number_of_edges()}.\n"
        f"Сущности:\n{entities or '- нет'}\n"
        f"Связи (вес – число документов, где связь встретилась):\n{relations or '- нет'}\n"
        f"Отказы узлов по числу отчётов:\n{components or '- нет'}\n"
        f"Отказы моделей по числу отчётов:\n{models or '- нет'}\n"
        f"Цепочки замен деталей:\n{replacements or '- нет'}\n"
        f"Детали сообщества по линейкам (по руководствам всего графа):\n{by_line or '- нет'}\n"
    )
    reply = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
    summary = " ".join(reply.content.split()[:MAX_WORDS])
    if not summary:
        fallback = " ".join(f"{kind}: {', '.join(names)}" for kind, names in sorted(types.items()))
        summary = " ".join(fallback.split()[:MAX_WORDS])
    return summary