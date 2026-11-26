"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes, normalize_code
from assistant.infra.llm import LLMClient, ToolCall, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

REFUSAL_TEXT = "В базе знаний не найдено."

SYSTEM_PROMPT = (
    "Ты ассистент сервисной службы «ПолярТех». Помогаешь диспетчеру найти нужную деталь, "
    "проверить её цену и наличие на складах в ERP и оформить заказ.\n"
    "Правила работы:\n"
    "1. Если в вопросе нет номера детали, сначала вызови search_knowledge (или summarize_topic для обзорных вопросов).\n"
    "2. Никогда не выдумывай коды деталей, цены и остатки.\n"
    "3. Различай исходы ERP: остаток 0 шт. означает, что деталь есть в каталоге, но закончилась; "
    "ошибка «не найдена» означает, что такой детали в ERP не существует.\n"
    "4. Если данных в базе знаний нет (refused=True), ответь: «В базе знаний не найдено.»\n"
    "5. Отвечай кратко, точно, с указанием кодов и чисел из результатов инструментов."
)

_NUMBER_RE = r"(?<![\w-])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?![\w-])"
_WITH_UNIT_RE = re.compile(rf"({_NUMBER_RE})\s*(руб|₽|шт)")


def _digits(raw: str) -> str:
    return re.sub(r"\D", "", raw)


def _extract_allowed_codes(text: str) -> set[str]:
    return set(find_codes(text))


def _has_user_quantity(question: str) -> bool:
    """Проверяет, назвал ли пользователь количество штук в вопросе."""
    q_without_codes = question
    for code in find_codes(question):
        q_without_codes = q_without_codes.replace(code, " ")
    # Убираем стандартные записи кодов с пробелами/дефисами, чтобы цифры внутри кодов не считались количеством
    q_cleaned = re.sub(r"\b[A-Za-zА-Яа-яЁё]{1,2}[- ]?\d{2,4}[A-Za-zА-Яа-яЁё]?\b", " ", q_without_codes)
    return bool(re.search(r"(?<!\d)\d+(?!\d)", q_cleaned))


def _verify_final_answer(answer: str, allowed_codes: set[str], tool_outputs: list[dict]) -> list[str]:
    """Возвращает список расхождений (неподтверждённых кодов, цен или остатков) в итоговом ответе."""
    problems: list[str] = []
    for code in find_codes(answer):
        if code not in allowed_codes:
            problems.append(f"код {code}")

    tool_text = json.dumps(tool_outputs, ensure_ascii=False)
    tool_numbers = set(re.findall(r"\d+", tool_text))
    for raw_num, unit in _WITH_UNIT_RE.findall(answer):
        num = _digits(raw_num)
        if num not in tool_numbers:
            problems.append(f"{num} {unit}")

    return problems


def _execute_with_retries(fn: Callable[..., dict], args: dict) -> tuple[dict | None, str | None]:
    """Вызывает инструмент; при ErpUnavailable повторяет ещё 2 раза (всего 3 попытки)."""
    last_exc: Exception | None = None
    for _ in range(3):
        try:
            res = fn(**args)
            if isinstance(res, dict) and "error" in res:
                return res, str(res["error"])
            return res, None
        except ErpUnavailable as exc:
            last_exc = exc
        except TypeError as exc:
            return None, f"неверные аргументы: {exc}"
    return None, f"ERP временно недоступна после 3 попыток: {last_exc}"


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение»."""
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    steps: list[ToolStep] = []
    sources: list[str] = []
    seen_sources: set[str] = set()
    allowed_codes: set[str] = _extract_allowed_codes(question)
    tool_outputs: list[dict] = []
    call_cache: dict[tuple[str, str], tuple[dict | None, str | None]] = {}



    def add_sources(new_sources: list[str]) -> None:
        for doc in new_sources:
            if doc not in seen_sources:
                seen_sources.add(doc)
                sources.append(doc)

    tool_calls_made = 0
    final_text = ""

    while True:
        available_schemas = TOOL_SCHEMAS if tool_calls_made < max_steps else None
        reply = llm.chat(messages, tools=available_schemas)
        messages.append(reply.as_message())

        if not reply.tool_calls:
            final_text = reply.content.strip()
            break

        for call in reply.tool_calls:
            if tool_calls_made >= max_steps:
                err = f"достигнут лимит шагов ({max_steps})"
                steps.append(ToolStep(call.name, call.arguments, error=err))
                messages.append(tool_message(call, {"error": err}))
                continue

            tool_calls_made += 1

            if call.name not in tools:
                err = f"неизвестный инструмент: {call.name}"
                steps.append(ToolStep(call.name, call.arguments, error=err))
                messages.append(tool_message(call, {"error": err}))
                continue

            if "_raw" in call.arguments:
                err = f"некорректный JSON в аргументах: {call.arguments['_raw']}"
                steps.append(ToolStep(call.name, call.arguments, error=err))
                messages.append(tool_message(call, {"error": err}))
                continue

            args = dict(call.arguments)

            if "part_id" in args:
                raw_part = str(args["part_id"])
                norm_part = normalize_code(raw_part) or raw_part.strip().upper()
                if norm_part not in allowed_codes:
                    err = (
                        f"номер детали {raw_part} не упоминался в вопросе и не найден в базе знаний; "
                        "сначала вызовите search_knowledge, чтобы узнать номер детали"
                    )
                    steps.append(ToolStep(call.name, args, error=err))
                    messages.append(tool_message(call, {"error": err}))
                    continue

            if call.name == "create_service_order":
                if not _has_user_quantity(question):
                    clarification = "Уточните, пожалуйста, какое количество деталей необходимо заказать?"
                    steps.append(ToolStep(call.name, args, error="количество не указано в вопросе"))
                    return AgentResult(
                        answer=clarification,
                        steps=steps,
                        sources=sources,
                        refused=False,
                        needs_confirmation=False,
                        pending_action=None,
                    )
                if not confirm:
                    return AgentResult(
                        answer="Подтвердите оформление заказа.",
                        steps=steps,
                        sources=sources,
                        refused=False,
                        needs_confirmation=True,
                        pending_action={"tool": "create_service_order", "args": args},
                    )

            cache_key = (call.name, json.dumps(args, ensure_ascii=False, sort_keys=True))
            if cache_key in call_cache:
                result, error = call_cache[cache_key]
            else:
                result, error = _execute_with_retries(tools[call.name], args)
                call_cache[cache_key] = (result, error)

            steps.append(ToolStep(call.name, args, result=result, error=error))


            if error is not None:
                payload = result if isinstance(result, dict) else {"error": error}
                messages.append(tool_message(call, payload))
            else:
                assert result is not None
                tool_outputs.append(result)
                allowed_codes |= _extract_allowed_codes(json.dumps(result, ensure_ascii=False))
                if isinstance(result.get("sources"), list):
                    add_sources(result["sources"])
                messages.append(tool_message(call, result))

                # Если лимит шагов исчерпан, просим модель подвести итог по уже собранным данным
        if tool_calls_made >= max_steps:
            messages.append(
                {
                    "role": "user",
                    "content": "Лимит вызовов инструментов исчерпан. Кратко ответь по уже полученным данным "
                    "и укажи, что проверка прервана из-за ограничения шагов.",
                }
            )
            final_reply = llm.chat(messages)
            final_text = final_reply.content.strip()
            if not final_text:
                gathered = "; ".join(
                    f"{s.tool}: {json.dumps(s.result, ensure_ascii=False)}"
                    for s in steps
                    if s.error is None and s.result is not None
                )
                final_text = f"Работа прервана по лимиту шагов ({max_steps}). Собранные данные: {gathered or 'нет'}."
            break
    
    if "не найдено" in final_text.lower():
        return AgentResult(answer=REFUSAL_TEXT, steps=steps, sources=[], refused=True)

    # 6. Финальная сверка кодов, цен и остатков в ответе (FR-10)
    problems = _verify_final_answer(final_text, allowed_codes, tool_outputs)
    if problems:
        messages.append(
            {
                "role": "user",
                "content": (
                    f"В твоём ответе есть неподтверждённые данные ({', '.join(problems)}). "
                    "Исправь ответ, используя ТОЛЬКО коды, цены и остатки из вопроса и результатов инструментов. "
                    "Если данных нет, напиши: «В базе знаний не найдено.»"
                ),
            }
        )
        retry_reply = llm.chat(messages)
        final_text = retry_reply.content.strip()
        if "не найдено" in final_text.lower():
            return AgentResult(answer=REFUSAL_TEXT, steps=steps, sources=[], refused=True)

        if _verify_final_answer(final_text, allowed_codes, tool_outputs):
            return AgentResult(answer=REFUSAL_TEXT, steps=steps, sources=[], refused=True)

    return AgentResult(
        answer=final_text,
        steps=steps,
        sources=sources,
        refused=False,
    )
