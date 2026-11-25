"""Сообщества графа и их сводки. Требование FR-3."""

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

SEED = 7
MAX_WORDS = 150
SYSTEM = (
    "Ты аналитик сервисной службы «ПолярТех». Напиши сводку сообщества графа знаний не длиннее 150 слов: "
    "какие сущности входят, как связаны, что встречается чаще. "
    "Используй только приведённые факты и числа, не выдумывай. Пиши по-русски."
)


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    # TODO: снять направление рёбер, разбить с учётом weight, вернуть Community с номерами по порядку
    raise NotImplementedError("Реализуйте detect_communities")


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    # TODO: собрать факты и подсчёты сообщества, попросить модель написать сводку
    sub_graph = graph.subgraph(community.nodes)

    types: dict[str, list[str]] = {}

    for node in sorted(sub_graph.nodes):
        types.setdefault(sub_graph.nodes[node]["type"], []).append(sub_graph.nodes[node]["name"])
    entities = "\n".join(f"- {kind}: {','.join(names)}" for kind, names in sorted(types.items()))

    edges = sorted(sub_graph.edges(data=True), key=lambda item: (-item[2].get("weight", 1), item[0], item[1]))
    relations = "\n".join(
        f"- {sub_graph.nodes[u]['name']} -{data['type']}-> {sub_graph.nodes[v]['name']} (вес {data.get('weight', 1)})"
        for u, v, data in edges[:80]
    )

    failures: dict[str, int] = {}

    for _, target, data in sub_graph.edges(data=True):
        if data["type"] == "FAILED":
            name = sub_graph.nodes[target]["name"]
            failures[name] = failures.get(name, 0) + data.get("weight", 1)
    totals = "\n".join(
        f"- {name}: {count}" for name, count in sorted(failures.items(), key=lambda item: (-item[1], item[0]))
    )

    prompt = (
        f"Сообщество {community.community_id}: узлов {sub_graph.number_of_nodes()}, связей {sub_graph.number_of_edges()}.\n"
        f"Сущности:\n{entities or '- нет'}\n"
        f"Связи (вес – число документов, где связь встретилась):\n{relations or '- нет'}\n"
        f"Отказы узлов по числу отчётов (сумма весов FAILED):\n{totals or '- нет'}\n"
    )

    reply = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
    summary = " ".join(reply.content.split()[:MAX_WORDS])
    if not summary:
        fallback = " ".join(f"{kind}: {', '.join(names)}" for kind, names in sorted(types.items()))
        summary = " ".join(fallback.split()[:MAX_WORDS])
    return summary
