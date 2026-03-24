from __future__ import annotations

import json
import logging
import os
from typing import Any

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover - optional dependency during bootstrap
    ChatGroq = None  # type: ignore[assignment]
    AIMessage = HumanMessage = SystemMessage = None  # type: ignore[assignment]

from .models import SafetyScoreResponse, StoredChatMessage

log = logging.getLogger(__name__)


def _normalize_temperature(value: str | float | None) -> float:
    default = 0.2
    if value is None:
        return default

    try:
        temperature = float(value)
    except (TypeError, ValueError):
        return default

    return min(max(temperature, 0.0), 2.0)


def _mode_guidance(mode: str) -> str:
    if mode == "map":
        return (
            "The user is looking at a live map. Mention distance, ETA, and the selected contact "
            "if that context is available. Keep the reply short and action-oriented."
        )

    if mode == "contacts":
        return (
            "The user is viewing emergency contacts. Help them understand who each contact is "
            "for and what to do next. Keep it short and clear."
        )

    if mode == "device":
        return (
            "The user is interacting with a device or alert flow. Focus on immediate safety "
            "steps, the device event, and what the user should do next."
        )

    return (
        "The user is the SafeHer User on this device. Assist only this user; do not provide "
        "support for other users, contacts, or third parties. Be calm, empathetic, and "
        "practical. When the threat level is high, lead with urgent safety actions."
    )


def _to_langchain_message(message: StoredChatMessage | Any) -> Any:
    role = getattr(message, "role", "user")
    content = getattr(message, "content", "")

    if role == "assistant" and AIMessage is not None:
        return AIMessage(content=content)
    if role == "system" and SystemMessage is not None:
        return SystemMessage(content=content)
    if HumanMessage is not None:
        return HumanMessage(content=content)
    return {"role": role, "content": content}


class GroqReplyGenerator:
    def __init__(
        self,
        *,
        model: str | None = None,
        temperature: str | float | None = None,
    ) -> None:
        self.model = model or os.getenv("SAFEHER_GROQ_MODEL", "llama-3.3-70b-versatile")
        self.temperature = _normalize_temperature(
            temperature if temperature is not None else os.getenv("SAFEHER_GROQ_TEMPERATURE")
        )
        api_key = os.getenv("GROQ_API_KEY")
        self._client = (
            ChatGroq(
                model=self.model,
                temperature=self.temperature,
                max_tokens=220,
                max_retries=2,
            )
            if ChatGroq is not None and api_key
            else None
        )

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate_reply(
        self,
        *,
        mode: str,
        analysis: SafetyScoreResponse,
        contacts: list[str],
        message: str,
        history: list[StoredChatMessage] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        if self._client is None:
            raise RuntimeError("Groq API key is not configured or langchain-groq is missing.")

        recent_history = list(history or [])[-8:]
        prompt_payload = {
            "mode": mode,
            "guidance": _mode_guidance(mode),
            "assistant_scope": "SafeHer User only; do not assist other users or contacts.",
            "latest_user_message": message,
            "analysis": analysis.model_dump(mode="json"),
            "trusted_contacts": contacts,
            "recent_history": [
                {"role": item.role, "content": item.content} for item in recent_history
            ],
            "context": context or {},
        }

        instructions = (
            "You are safeher-ai, a safety-first AI assistant for the SafeHer User only. "
            "Do not provide support for other users, contacts, or third parties. "
            "Write a concise, reassuring, and practical reply. "
            "Use plain language and avoid markdown unless it improves readability. "
            "Keep the reply to 1-4 short paragraphs. "
            "Do not invent facts that are not in the prompt. "
            "If the analysis is high risk, be direct about urgent safety steps."
        )

        prompt_messages: list[Any] = [
            SystemMessage(content=instructions)
            if SystemMessage is not None
            else {"role": "system", "content": instructions},
            SystemMessage(
                content=(
                    "Structured context:\n"
                    f"{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}"
                )
            )
            if SystemMessage is not None
            else {
                "role": "system",
                "content": json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            },
        ]

        prompt_messages.extend(_to_langchain_message(item) for item in recent_history)

        if (
            not recent_history
            or getattr(recent_history[-1], "role", None) != "user"
            or getattr(recent_history[-1], "content", None) != message
        ):
            prompt_messages.append(
                HumanMessage(content=message)
                if HumanMessage is not None
                else {"role": "user", "content": message}
            )

        response = self._client.invoke(prompt_messages)

        reply = getattr(response, "content", "")
        if isinstance(reply, list):
            reply = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part) for part in reply
            )

        reply = str(reply).strip()
        if not reply:
            raise RuntimeError("Groq returned an empty reply.")
        return reply


OpenAIReplyGenerator = GroqReplyGenerator
