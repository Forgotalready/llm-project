"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from pydantic import BaseModel, ValidationError

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes, find_names
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

MAX_TOKENS = 4096

RETRY_NOTE = "Ответ не соответствует схеме. Верни только корректный JSON: entities и relations."

SYSTEM_PROMPT = "\n".join(
    [
        "Ты извлекаешь сущности и связи из документации «ПолярТех» строго по схеме.",
        "",
        "Типы сущностей:",
        "- ProductLine: линейка (Арктика, Бриз, Гольфстрим)",
        "- Model: код модели (AR-210, BZ-110, GS-410)",
        "- Component: узел строчными буквами (пусковое реле, компрессор)",
        "- Part: номер детали (P-1042, P-1042B)",
        "- ErrorCode: код ошибки (E-17)",
        "- Procedure: номер регламента (SP-104)",
        "- Supplier: поставщик (ХолодКомплект)",
        "",
        "Связи и их СТРОГОЕ направление:",
        "1. Model -> BELONGS_TO -> ProductLine",
        "2. Model -> HAS_PART -> Part",
        "3. Part -> IS_A -> Component",
        "4. ErrorCode -> INDICATES -> Component",
        "5. ErrorCode -> FIXED_BY -> Procedure",
        "6. Part -> REPLACED_BY -> Part (старая снятая -> новая)",
        "7. Part -> SUPPLIED_BY -> Supplier",
        "8. Model -> FAILED -> Component",
        "",
        "Правила:",
        "- Используй только то, что явно сказано в тексте; коды и названия не выдумывай.",
        "- Названия пиши в словарной форме, в именительном падеже.",
        "- Пройди по ВСЕМ строкам списков в тексте, не пропускай ни одной.",
        "- Верни компактный JSON без лишних пробелов и переносов.",
        "",
        "Пример.",
        'Текст: "Деталь P-1042 (пусковое реле) снята с производства.',
        'Вместо неё устанавливается деталь P-1042B. Поставщик P-1042B – «ХолодКомплект»."',
        "Ответ:",
        '{"entities": [{"type": "Part", "name": "P-1042"}, {"type": "Part", "name": "P-1042B"},',
        '{"type": "Component", "name": "пусковое реле"}, {"type": "Supplier", "name": "ХолодКомплект"}],',
        '"relations": [{"subject_type": "Part", "subject": "P-1042", "type": "REPLACED_BY",',
        '"object_type": "Part", "object": "P-1042B"},',
        '{"subject_type": "Part", "subject": "P-1042", "type": "IS_A",',
        '"object_type": "Component", "object": "пусковое реле"},',
        '{"subject_type": "Part", "subject": "P-1042B", "type": "SUPPLIED_BY",',
        '"object_type": "Supplier", "object": "ХолодКомплект"}]}',
    ]
)

# Подсказка по виду документа: модель пропускает связи, которые в тексте есть
# (потеряны 5 FIXED_BY и 4 SUPPLIED_BY), и домысливает линейку в отчётах.
DOC_HINTS = {
    "manual": (
        "Это руководство по модели. Из первого абзаца верни BELONGS_TO (модель -> линейка). "
        "Для КАЖДОЙ строки раздела «Состав узлов» верни три связи: HAS_PART (модель -> деталь), "
        "IS_A (деталь -> узел), SUPPLIED_BY (деталь -> поставщик). "
        "Для каждой строки раздела «Коды ошибок» верни INDICATES (код -> узел). "
        "Обычно это 8 узлов и 10 кодов — не пропускай строки."
    ),
    "bulletin": (
        "Это сервисный бюллетень. Если деталь снята с производства, верни REPLACED_BY (старая -> новая), "
        "IS_A для обеих деталей и SUPPLIED_BY для названной детали. "
        "Если сказано, что деталь не снята и не заменяется, связи REPLACED_BY быть не должно."
    ),
    "procedure": (
        "Это регламент ремонта. Верни FIXED_BY (код ошибки -> номер регламента) и INDICATES (код -> узел). "
        "Если регламент относится к плановым работам и код ошибки не назван, верни пустые списки."
    ),
    "report": (
        "Это отчёт о выезде. Верни только FAILED (модель -> отказавший узел). "
        "Линейку, номера деталей и поставщиков не домысливай: в отчёте их нет."
    ),
}


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


def _strip_fence(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    text = text[3:]
    if text[:4].lower() == "json":
        text = text[4:]
    return text.split("```", 1)[0].strip()


def _close_truncated(text: str) -> str:
    """Отрезает оборванный хвост и закрывает скобки: страховка к лимиту токенов."""
    cut = text.rfind("}")
    if cut == -1:
        return text
    head = text[: cut + 1]
    stack: list[str] = []
    in_string = escaped = False
    for char in head:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]" and stack:
            stack.pop()
    return head + "".join("}" if opener == "{" else "]" for opener in reversed(stack))


def _parse(raw: str) -> _RawExtraction | None:
    text = _strip_fence(raw)
    for candidate in (text, _close_truncated(text)):
        try:
            return _RawExtraction.model_validate(json.loads(candidate))
        except (json.JSONDecodeError, ValidationError):
            continue
    return None


def _user_prompt(chunk: Chunk) -> str:
    hint = DOC_HINTS.get(chunk.doc_id.split("_", 1)[0], "")
    return f"{hint}\n\nТекст чанка:\n{chunk.text}" if hint else f"Текст чанка:\n{chunk.text}"


def _validate_entity(
    entity_type: str,
    raw_name: str,
    chunk_codes: set[str],
    chunk_names: set[tuple[str, str]],
    vocab: dict[str, list[str]],
) -> Entity | None:
    if entity_type not in ENTITY_TYPES:
        return None
    canon = canonical_name(entity_type, raw_name, vocab)
    if canon is None:
        return None

    if entity_type in CODE_TYPES:
        return Entity(entity_type, canon) if canon in chunk_codes else None
    return Entity(entity_type, canon) if (entity_type, canon) in chunk_names else None


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(chunk)},
    ]

    parsed: _RawExtraction | None = None
    for attempt in range(2):
        response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA, max_tokens=MAX_TOKENS)
        parsed = _parse(response.content)
        if parsed is not None:
            break
        if attempt == 0:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": RETRY_NOTE})

    if parsed is None:
        return ExtractionResult(chunk_id=chunk.chunk_id, doc_id=chunk.doc_id)

    vocab = vocabulary()
    chunk_codes = set(find_codes(chunk.text))
    chunk_names = set(find_names(chunk.text, vocab))
    dropped = 0

    entities_by_id: dict[str, Entity] = {}
    for item in parsed.entities:
        try:
            raw_ent = _RawEntity.model_validate(item)
        except ValidationError:
            dropped += 1
            continue
        ent = _validate_entity(raw_ent.type, raw_ent.name, chunk_codes, chunk_names, vocab)
        if ent is None:
            dropped += 1
        else:
            entities_by_id[ent.node_id] = ent

    relations_by_triple: dict[tuple[str, str, str], Relation] = {}
    for item in parsed.relations:
        try:
            raw_rel = _RawRelation.model_validate(item)
        except ValidationError:
            dropped += 1
            continue
        if raw_rel.type not in RELATION_SIGNATURES:
            dropped += 1
            continue
        subj = _validate_entity(raw_rel.subject_type, raw_rel.subject, chunk_codes, chunk_names, vocab)
        obj = _validate_entity(raw_rel.object_type, raw_rel.object, chunk_codes, chunk_names, vocab)
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