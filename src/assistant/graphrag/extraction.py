"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json
from pydantic import BaseModel, ValidationError

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    ENTITY_TYPES,
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

SYSTEM_PROMPT = """Ты извлекаешь сущности и связи из технической документации «ПолярТех» строго по схеме.

Типы сущностей (ENTITY_TYPES):
- ProductLine: линейка оборудования (Арктика, Бриз, Гольфстрим)
- Model: код модели (например, AR-210, BZ-110, GS-410)
- Component: название узла строчными буквами (например, пусковое реле, компрессор)
- Part: номер детали (например, P-1042, P-1042B)
- ErrorCode: код ошибки (например, E-17)
- Procedure: номер регламента (например, SP-104)
- Supplier: название поставщика (например, ХолодКомплект)

Допустимые связи и их СТРОГОЕ направление (subject_type -> TYPE -> object_type):
1. Model -> BELONGS_TO -> ProductLine (модель входит в линейку)
2. Model -> HAS_PART -> Part (в модели установлена деталь)
3. Part -> IS_A -> Component (деталь является узлом этого вида)
4. ErrorCode -> INDICATES -> Component (ошибка указывает на отказ узла; извлекай из руководств и регламентов, но НЕ из отчётов о выезде)
5. ErrorCode -> FIXED_BY -> Procedure (ошибка устраняется по регламенту)
6. Part -> REPLACED_BY -> Part (старая снятая деталь -> новая деталь на замену)
7. Part -> SUPPLIED_BY -> Supplier (деталь поставляет поставщик)
8. Model -> FAILED -> Component (в отчёте о выезде зафиксирован отказ узла у модели)

Правила:
- Используй только сущности и коды, явно упомянутые в тексте. Не выдумывай коды и названия.
- Для названий (ProductLine, Component, Supplier) используй словарную форму в именительном падеже.

Пример:
Текст: "Деталь P-1042 (пусковое реле) снята с производства. Вместо неё устанавливается деталь P-1042B. Поставщик детали P-1042B – «ХолодКомплект»."
Ответ:
{
  "entities": [
    {"type": "Part", "name": "P-1042"},
    {"type": "Part", "name": "P-1042B"},
    {"type": "Component", "name": "пусковое реле"},
    {"type": "Supplier", "name": "ХолодКомплект"}
  ],
  "relations": [
    {"subject_type": "Part", "subject": "P-1042", "type": "REPLACED_BY", "object_type": "Part", "object": "P-1042B"},
    {"subject_type": "Part", "subject": "P-1042", "type": "IS_A", "object_type": "Component", "object": "пусковое реле"},
    {"subject_type": "Part", "subject": "P-1042B", "type": "SUPPLIED_BY", "object_type": "Supplier", "object": "ХолодКомплект"}
  ]
}"""


class _RawEntity(BaseModel):
    type: str
    name: str


class _RawRelation(BaseModel):
    subject_type: str
    subject: str
    type: str
    object_type: str
    object: str


class _RawExtraction(BaseModel):
    entities: list[_RawEntity]
    relations: list[_RawRelation]


def _validate_entity(
    entity_type: str, raw_name: str, chunk_codes: set[str], vocab: dict[str, list[str]]
) -> Entity | None:
    if entity_type not in ENTITY_TYPES:
        return None
    canon = canonical_name(entity_type, raw_name, vocab)
    if canon is None:
        return None
    if entity_type in CODE_TYPES and canon not in chunk_codes:
        return None
    return Entity(entity_type, canon)


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": chunk.text},
    ]

    parsed: _RawExtraction | None = None
    for _ in range(2):
        response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA)
        try:
            data = json.loads(response.content)
            parsed = _RawExtraction.model_validate(data)
            break
        except (json.JSONDecodeError, ValidationError):
            continue

    if parsed is None:
        return ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)

    vocab = vocabulary()
    chunk_codes = set(find_codes(chunk.text))
    dropped = 0

    entities_by_id: dict[str, Entity] = {}
    for raw_ent in parsed.entities:
        ent = _validate_entity(raw_ent.type, raw_ent.name, chunk_codes, vocab)
        if ent is None:
            dropped += 1
        else:
            entities_by_id[ent.node_id] = ent

    relations_by_triple: dict[tuple[str, str, str], Relation] = {}
    for raw_rel in parsed.relations:
        if raw_rel.type not in RELATION_SIGNATURES:
            dropped += 1
            continue
        subj = _validate_entity(raw_rel.subject_type, raw_rel.subject, chunk_codes, vocab)
        obj = _validate_entity(raw_rel.object_type, raw_rel.object, chunk_codes, vocab)
        if subj is None or obj is None:
            dropped += 1
            continue
        rel = Relation(subj, raw_rel.type, obj)
        if not rel.matches_schema():
            dropped += 1
            continue
        entities_by_id[subj.node_id] = subj
        entities_by_id[obj.node_id] = obj
        relations_by_triple[(subj.node_id, rel.type, obj.node_id)] = rel

    return ExtractionResult(
        chunk_id=chunk.chunk_id,
        doc_id=chunk.doc_id,
        entities=list(entities_by_id.values()),
        relations=list(relations_by_triple.values()),
        dropped=dropped,
    )
