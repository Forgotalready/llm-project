"""Инструменты агента. Требование FR-6."""

from collections.abc import Callable

from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase
from assistant.infra.grounding import normalize_code, code_type
from assistant.graphrag.search import global_search, local_search

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
# TODO: заполните список. У параметра warehouse четыре склада задаются перечислением.

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Поиск по базе знаний (граф документов): модель, ошибка, узел, деталь, замена, регламент.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "Вопрос к базе знаний."}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": "Обзорный ответ по всему корпусу (например, какой узел чаще отказывает в линейке).",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string", "description": "Обзорный вопрос."}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": "Остаток детали на складе и признак discontinued. Остаток 0 и «данных нет» – разные исходы.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "Номер детали вида P-1042 или P-1042B."},
                    "warehouse": {"type": "string", "enum": WAREHOUSES, "description": "Склад."},
                },
                "required": ["part_id", "warehouse"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Цена детали в рублях и признак discontinued.",
            "parameters": {
                "type": "object",
                "properties": {"part_id": {"type": "string", "description": "Номер детали вида P-1042 или P-1042B."}},
                "required": ["part_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_service_order",
            "description": "Создать заказ детали. Выполняется только после подтверждения человека.",
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {"type": "string", "description": "Номер детали вида P-1042 или P-1042B."},
                    "warehouse": {"type": "string", "enum": WAREHOUSES, "description": "Склад."},
                    "qty": {"type": "integer", "minimum": 1, "description": "Количество, названное пользователем."},
                    "model": {"type": "string", "description": "Код модели, например AR-210."},
                },
                "required": ["part_id", "warehouse", "qty", "model"],
            },
        },
    },
]


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    # TODO: пять функций поверх local_search, global_search и клиента ERP
    def part(raw: object) -> tuple[str | None, str | None]:
        code = normalize_code(raw) if isinstance(raw, str) else None
        if code is None or code_type(code) != "Part":
            return None, f"неверный номер детали {raw!r}: ожидается вид P-1042 или P-1042B"
        return code, None

    def warehouse_check(raw: object) -> str | None:
        if raw not in WAREHOUSES:
            return f"неизвестный склад {raw!r}, допустимые: {', '.join(WAREHOUSES)}"
        return None

    def search_knowledge(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "пустой вопрос"}

        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize_topic(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "пустой вопрос"}
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        code, error = part(part_id)
        error = error or warehouse_check(warehouse)

        if error:
            return {"error": error}

        try:
            return erp.get_stock(code, warehouse)
        except ErpNotFound as exc:
            return {"error": f"данных нет: {exc}"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила запрос: {exc}"}

    def get_price(part_id: str) -> dict:
        code, error = part(part_id)
        if error:
            return {"error": error}
        try:
            return erp.get_price(code)
        except ErpNotFound as exc:
            return {"error": f"данных нет: {exc}"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила запрос: {exc}"}

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        code, error = part(part_id)
        error = error or warehouse_check(warehouse)
        if error:
            return {"error": error}
        if isinstance(qty, str) and qty.isdigit():
            qty = int(qty)
        if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
            return {"error": f"неверное количество {qty!r}: нужно целое больше нуля"}
        model_code = normalize_code(model) if isinstance(model, str) else None
        if model_code is None or code_type(model_code) != "Model":
            return {"error": f"неверный код модели {model!r}"}
        try:
            return erp.create_order(code, warehouse, qty, model_code)
        except ErpNotFound as exc:
            return {"error": f"данных нет: {exc}"}
        except ErpRejected as exc:
            return {"error": f"ERP отклонила запрос: {exc}"}

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }