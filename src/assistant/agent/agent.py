"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import code_type, find_codes, normalize_code
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

SYSTEM = (
    "Ты ассистент диспетчера сервисной службы «ПолярТех». Отвечай по-русски. "
    "Инструменты: search_knowledge (граф документов), summarize_topic (обзор), check_stock, get_price, "
    "create_service_order (ERP). Правила: номера деталей бери только из вопроса или результатов "
    "search_knowledge; сначала найди деталь в базе знаний, затем обращайся в ERP; остаток 0 и «данных нет» "
    "– разные исходы; цены и остатки называй только из результатов инструментов; заказ создаётся только "
    "после подтверждения человека; если данных нет, напиши «в базе знаний не найдено»."
)
REFUSAL = "В базе знаний не найдено."
# Обязательные аргументы берём из тех же схем, что видит модель: один источник правды.
REQUIRED = {
    schema["function"]["name"]: set(schema["function"]["parameters"].get("required", [])) for schema in TOOL_SCHEMAS
}
ERP_TOOLS = {"check_stock", "get_price", "create_service_order"}
# Число, стоящее отдельно: «1042» внутри P-1042B числом не считается.
_PLAIN_NUMBER = re.compile(r"(?<![\w-])\d+(?![\w-])")
# Проверяем только цены и остатки (руб/₽/шт). Норму времени в минутах не трогаем:
# она есть в документах и выдуманной ценой не является.
_UNIT_NUMBER = re.compile(r"(?<![\w-])(\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)\s*(руб|₽|шт)")


def _invoke(fn: Callable[..., dict], args: dict) -> tuple[dict | None, str | None]:
    """Вызов инструмента с повторами только на «сервис недоступен»."""
    last = None
    for _ in range(3):  # FR-8: первая попытка плюс два повтора
        try:
            return fn(**args), None
        except ErpUnavailable as exc:
            last = str(exc)  # временный сбой — есть смысл повторить
        except TypeError as exc:
            # Лишний или неверный аргумент: повтор ничего не изменит, отдаём ошибку модели.
            return None, f"аргументы не подходят инструменту: {exc}"
    return None, f"данные ERP получить не удалось: {last}"


def _fabricated(text: str, question: str, tool_text: str) -> list[str]:
    """Что в ответе взято не из вопроса и не из результатов инструментов (FR-10)."""
    allowed_codes = set(find_codes(question)) | set(find_codes(tool_text))
    issues = [code for code in find_codes(text) if code not in allowed_codes]
    allowed_numbers = set(re.findall(r"\d+", tool_text)) | set(re.findall(r"\d+", question))
    for raw, unit in _UNIT_NUMBER.findall(text):
        digits = re.sub(r"\D", "", raw)  # «2 630» и «2630» — одно число
        if digits not in allowed_numbers:
            issues.append(f"{digits} {unit}")
    return issues


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение»."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    steps: list[ToolStep] = []
    sources: list[str] = []
    cache: dict[str, dict] = {}  # FR-7: повторный вызов с теми же аргументами не идёт в ERP снова
    # Номер детали разрешено брать только отсюда: вопрос плюс результаты прошлых инструментов.
    allowed_parts = {code for code in find_codes(question) if code_type(code) == "Part"}
    question_numbers = set(_PLAIN_NUMBER.findall(question))
    executed = 0
    interrupted = False
    final_text = ""

    # Запас итераций сверх max_steps нужен, чтобы после лимита модель успела дать итоговый ответ.
    for _ in range(max_steps + 8):
        reply = llm.chat(messages, tools=TOOL_SCHEMAS)
        if not reply.tool_calls:
            final_text = reply.content  # модель ответила текстом — цикл закончен
            break
        messages.append(reply.as_message())

        if executed >= max_steps:
            # Инструменты больше не выполняем, но цикл не рвём: каждое обращение модели
            # получает наблюдение с ошибкой, пока она не сформулирует итог (FR-7).
            for call in reply.tool_calls:
                messages.append(tool_message(call, {"error": "лимит шагов исчерпан, инструмент не выполнен"}))
            if not interrupted:
                interrupted = True
                messages.append(
                    {
                        "role": "user",
                        "content": "Лимит шагов исчерпан. Дай итоговый ответ и сообщи, что работа прервана."
                    }
                )
            continue

        pending = None
        for call in reply.tool_calls:
            name, args = call.name, call.arguments
            error = None
            if name not in tools:
                error = f"неизвестный инструмент {name!r}, доступны: {', '.join(sorted(tools))}"
            elif not isinstance(args, dict) or "_raw" in args:
                # LLMClient кладёт "_raw", когда аргументы не разобрались как JSON.
                error = "аргументы вызова не читаются, передай корректный JSON по схеме инструмента"
            elif not REQUIRED.get(name, set()) <= set(args):
                missing = sorted(REQUIRED.get(name, set()) - set(args))
                error = f"не хватает аргументов: {', '.join(missing)}"

            if error is None and name in ERP_TOOLS:
                raw_part = args.get("part_id")
                part = normalize_code(raw_part) if isinstance(raw_part, str) else None
                if part is None or code_type(part) != "Part":
                    error = f"неверный номер детали {raw_part!r}"
                elif part not in allowed_parts:
                    # Главная защита от выдуманной детали: номер должен быть чем-то подтверждён.
                    error = (
                        f"номер детали {part} не встречается в вопросе или результатах инструментов. "
                        "Сначала вызови search_knowledge и возьми номер из его ответа."
                    )
                else:
                    args = {**args, "part_id": part}  # в ERP и в steps уходит канонический номер

            if error is None and name == "create_service_order":
                qty = args.get("qty")
                if isinstance(qty, str) and qty.isdigit():
                    qty = int(qty)
                if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
                    error = f"неверное количество {args.get('qty')!r}"
                elif str(qty) not in question_numbers:
                    # Количество разрешено брать ТОЛЬКО из вопроса: подставленная единица —
                    # это заказ, которого человек не просил.
                    error = "количество не названо пользователем, уточни его у диспетчера"
                else:
                    model_raw = args.get("model")
                    model = normalize_code(model_raw) if isinstance(model_raw, str) else None
                    if model is None or code_type(model) != "Model":
                        error = f"неверный код модели {model_raw!r}"
                    else:
                        args = {**args, "qty": qty, "model": model}

            if error is None and name == "create_service_order" and not confirm:
                # FR-9: заказ дошёл до вызова, но без подтверждения человека не выполняем.
                # Шаг в steps НЕ пишем: успешным считается шаг с error=None, а заказа не было.
                pending = {"tool": name, "args": args}
                messages.append(tool_message(call, {"status": "ожидает подтверждения человека", **args}))
                break

            if error is not None:
                # Отклонённый вызов — это наблюдение для модели, цикл продолжается.
                steps.append(ToolStep(name, args if isinstance(args, dict) else {}, error=error))
                messages.append(tool_message(call, {"error": error}))
                continue

            key = json.dumps({"tool": name, "args": args}, ensure_ascii=False, sort_keys=True)
            if key in cache:
                result = cache[key]  # тот же вопрос к ERP — тот же ответ, без сетевого вызова
            else:
                executed += 1  # лимит считаем по реальным выполнениям, а не по ошибкам проверок
                result, call_error = _invoke(tools[name], args)
                if call_error is not None:
                    steps.append(ToolStep(name, args, error=call_error))
                    messages.append(tool_message(call, {"error": call_error}))
                    continue
                cache[key] = result

            if "error" in result:
                # «Данных нет» и отказ ERP помечаем как ошибку шага, но текст сохраняем для модели:
                # исходы должны остаться различимыми и в ответе тоже.
                steps.append(ToolStep(name, args, error=str(result["error"])))
            else:
                steps.append(ToolStep(name, args, result=result))
                # Результат поиска расширяет список разрешённых номеров деталей:
                # так цепочка «граф -> ERP» становится возможной.
                for code in find_codes(json.dumps(result, ensure_ascii=False)):
                    if code_type(code) == "Part":
                        allowed_parts.add(code)
                for doc in result.get("sources") or []:
                    if doc not in sources:
                        sources.append(doc)  # источники копим по ходу диалога (FR-10)
            messages.append(tool_message(call, result))

        if pending is not None:
            # Ответ собираем кодом: он обязан назвать деталь, количество, склад и модель,
            # иначе человеку нечего подтверждать.
            a = pending["args"]
            answer = (
                f"Готов заказать {a['qty']} шт. детали {a['part_id']} со склада «{a['warehouse']}» "
                f"для {a['model']}. Подтвердите заказ."
            )
            return AgentResult(answer, steps=steps, sources=sources, needs_confirmation=True, pending_action=pending)

    text = final_text.strip()
    if interrupted:
        text = (text + " Работа прервана: лимит шагов исчерпан.").strip()
    if not text:
        return AgentResult(REFUSAL, steps=steps, sources=sources, refused=True)

    # FR-10: сверяем готовый ответ с фактами. В ход идут только успешные результаты инструментов.
    tool_text = json.dumps([step.result for step in steps if step.result is not None], ensure_ascii=False)
    issues = _fabricated(text, question, tool_text)
    if issues:
        messages.append({"role": "assistant", "content": text})
        messages.append(
            {
                "role": "user",
                "content": "В ответе есть факты, которых нет в вопросе и результатах инструментов: "
                + ", ".join(issues)
                + ". Исправь ответ, опирайся только на полученные данные.",
            }
        )
        reply = llm.chat(messages, tools=TOOL_SCHEMAS)  # ровно одна попытка исправления
        text = reply.content.strip()
        if reply.tool_calls or not text or _fabricated(text, question, tool_text):
            # Второй раз расходимся с фактами — отказываем фиксированной формулировкой:
            # в ней заведомо нет ни выдуманного кода, ни выдуманной цены.
            return AgentResult(REFUSAL, steps=steps, sources=sources, refused=True)

    return AgentResult(text, steps=steps, sources=sources, refused="не найдено" in text.lower())