"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult, Entity


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    # TODO: узлы, рёбра с весами, проверка схемы, разбор встречных замен
    node_meta: dict[str, Entity] = {}

    node_sources: dict[str, set[str]] = {}
    edge_chunks: dict[tuple[str, str, str], set[str]] = {}

    for result in results:
        for entity in result.entities:
            node_meta.setdefault(entity.node_id, entity)
            node_sources.setdefault(entity.node_id, set()).add(result.chunk_id)
        for relation in result.relations:
            if not relation.matches_schema():
                continue

            for end in (relation.subject, relation.object):
                node_meta.setdefault(end.node_id, end)
                node_sources.setdefault(end.node_id, set()).add(result.chunk_id)
            key = (relation.subject.node_id, relation.type, relation.object.node_id)
            edge_chunks.setdefault(key, set()).add(result.chunk_id)

    graph = nx.DiGraph(conflicts=0)
    for node_id, entity in node_meta.items():
        graph.add_node(node_id, type=entity.type, name=entity.name, sources=sorted(node_sources[node_id]))

    for (source, kind, target), chunks in edge_chunks.items():
        graph.add_edge(source, target, type=kind, weight=len(chunks), sources=sorted(chunks))
    conflicts = 0
    checked: set[frozenset[str]] = set()

    for u, v, data in list(graph.edges(data=True)):
        if data["type"] != "REPLACED_BY" or not graph.has_edge(v, u):
            continue
        if graph.edges[v, u]["type"] != "REPLACED_BY":
            continue
        pair = frozenset((u, v))
        if pair in checked:
            continue
        checked.add(pair)
        conflicts += 1

        forward = graph.edges[u, v]["weight"]
        backward = graph.edges[v, u]["weight"]

        if forward > backward:
            graph.remove_edge(v, u)
        elif backward > forward:
            graph.remove_edge(u, v)
        else:
            graph.remove_edge(u, v)
            graph.remove_edge(v, u)
    
    graph.graph['conflicts'] = conflicts
    return graph

    