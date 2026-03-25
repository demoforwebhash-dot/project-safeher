from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

from .models import SafetyScoreResponse


TOKEN_RE = re.compile(r"[a-z0-9']+")

LABELS = ("low", "medium", "high")

LABEL_SCORES = {
    "low": 0.15,
    "medium": 0.58,
    "high": 0.93,
}

LABEL_RECOMMENDED_ACTIONS = {
    "low": [],
    "medium": [
        "share_live_location",
        "call_emergency_contact",
        "find_safe_route",
    ],
    "high": [
        "trigger_sos",
        "share_live_location",
        "call_emergency_contact",
        "notify_authorities",
    ],
}

DEFAULT_TRAINING_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("hello", "low"),
    ("i am okay", "low"),
    ("thanks for helping", "low"),
    ("can you share my location", "low"),
    ("i feel worried", "medium"),
    ("this feels unsafe", "medium"),
    ("i am lost and alone", "medium"),
    ("i need a safe route", "medium"),
    ("help me", "high"),
    ("someone is following me", "high"),
    ("i am in danger", "high"),
    ("panic sos", "high"),
    ("attack", "high"),
    ("kidnap", "high"),
    ("abuse", "high"),
    ("trapped", "high"),
)

_SIGNAL_ALIASES = {
    "location_present": "location_available",
    "location_available": "location_available",
    "location_unavailable": "location_unavailable",
    "location_missing": "location_unavailable",
    "night_time": "night_time",
    "nighttime": "night_time",
    "night-time": "night_time",
    "network_offline": "network_offline",
    "offline": "network_offline",
    "battery_critical": "battery_critical",
    "battery_low": "battery_critical",
    "moving_fast": "moving_fast",
    "moving": "moving_fast",
    "fast": "moving_fast",
}

_SIGNAL_LABELS = {
    "location_available": "Location available",
    "location_unavailable": "Location unavailable",
    "night_time": "Night-time",
    "network_offline": "Network offline",
    "battery_critical": "Battery critical",
    "moving_fast": "Moving fast",
}


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


def _context_float(context: dict[str, Any], key: str) -> float | None:
    value = context.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    return None


def _context_bool(context: dict[str, Any], key: str) -> bool | None:
    value = context.get(key)
    if isinstance(value, bool):
        return value

    return None


def _context_str(context: dict[str, Any], key: str) -> str | None:
    value = context.get(key)
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None

    return None


def _context_str_list(context: dict[str, Any], key: str) -> list[str]:
    value = context.get(key)
    if not isinstance(value, list):
        return []

    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            normalized = item.strip()
            if normalized:
                result.append(normalized)

    return result


def _normalize_signal(signal: str) -> str | None:
    normalized = signal.strip().casefold().replace(" ", "_").replace("-", "_")
    return _SIGNAL_ALIASES.get(normalized)


def _softmax(log_values: dict[str, float]) -> dict[str, float]:
    if not log_values:
        return {label: 0.0 for label in LABELS}

    max_value = max(log_values.values())
    exp_values = {label: math.exp(value - max_value) for label, value in log_values.items()}
    total = sum(exp_values.values()) or 1.0
    return {label: value / total for label, value in exp_values.items()}


def _score_distribution(score: float) -> dict[str, float]:
    logits = {
        "low": 1.4 - 4.0 * score,
        "medium": 0.5 - abs(score - 0.5) * 3.0,
        "high": -1.2 + 4.0 * score,
    }
    probabilities = _softmax(logits)
    return {label: round(probabilities.get(label, 0.0), 4) for label in LABELS}


@dataclass
class _ModelState:
    class_counts: Counter[str]
    token_counts: dict[str, Counter[str]]
    token_totals: dict[str, int]
    vocabulary: set[str]


class SafetyRiskModel:
    def __init__(self, training_examples: Iterable[tuple[str, str]] | None = None) -> None:
        self._state = self._fit(training_examples or DEFAULT_TRAINING_EXAMPLES)
        self._high_tokens = {
            token
            for token, count in self._state.token_counts["high"].items()
            if count >= 1
        }
        self._medium_tokens = {
            token
            for token, count in self._state.token_counts["medium"].items()
            if count >= 1
        }

    def _fit(self, examples: Iterable[tuple[str, str]]) -> _ModelState:
        class_counts: Counter[str] = Counter()
        token_counts: dict[str, Counter[str]] = defaultdict(Counter)
        token_totals: dict[str, int] = defaultdict(int)
        vocabulary: set[str] = set()

        for text, label in examples:
            if label not in LABELS:
                continue

            class_counts[label] += 1
            tokens = _tokenize(text)
            vocabulary.update(tokens)
            token_counts[label].update(tokens)
            token_totals[label] += len(tokens)

        for label in LABELS:
            class_counts.setdefault(label, 1)
            token_counts.setdefault(label, Counter())
            token_totals.setdefault(label, 0)

        return _ModelState(
            class_counts=class_counts,
            token_counts=token_counts,
            token_totals=token_totals,
            vocabulary=vocabulary,
        )

    def _predict_probabilities(self, text: str) -> dict[str, float]:
        tokens = _tokenize(text)
        vocab_size = max(len(self._state.vocabulary), 1)
        total_examples = sum(self._state.class_counts.values()) or 1
        log_values: dict[str, float] = {}

        for label in LABELS:
            class_count = self._state.class_counts[label]
            prior = math.log(class_count / total_examples)
            token_total = self._state.token_totals[label]
            label_counts = self._state.token_counts[label]
            log_probability = prior

            for token in tokens:
                token_probability = (label_counts[token] + 1) / (token_total + vocab_size)
                log_probability += math.log(token_probability)

            log_values[label] = log_probability

        probabilities = _softmax(log_values)
        return {label: round(probabilities.get(label, 0.0), 4) for label in LABELS}

    def _text_score(self, text: str) -> float:
        probabilities = self._predict_probabilities(text)
        return sum(probabilities[label] * LABEL_SCORES[label] for label in LABELS)

    def _live_signal_context(
        self,
        *,
        location_present: bool,
        context: dict[str, Any],
        requested_signals: set[str],
    ) -> tuple[list[str], float]:
        factors: list[str] = []
        live_score = 0.12

        location_status = (_context_str(context, "location_status") or "").casefold()
        location_available = (
            location_present
            or location_status in {"available", "live", "watching", "location_available"}
            or "location_available" in requested_signals
        )
        location_unavailable = (
            location_status in {"unavailable", "denied", "unsupported", "location_unavailable"}
            or "location_unavailable" in requested_signals
        )

        if location_available:
            live_score -= 0.08
            factors.append(_SIGNAL_LABELS["location_available"])
        elif location_unavailable:
            live_score += 0.14
            factors.append(_SIGNAL_LABELS["location_unavailable"])

        gps_accuracy_m = _context_float(context, "gps_accuracy_m")
        if gps_accuracy_m is not None and gps_accuracy_m > 50:
            if gps_accuracy_m > 150:
                live_score += 0.06
            else:
                live_score += 0.03
            factors.append("GPS accuracy weak")

        speed_kmh = _context_float(context, "speed_kmh")
        movement_state = _context_str(context, "movement_state")
        moving_fast = (
            "moving_fast" in requested_signals
            or movement_state in {"fast", "moving"}
            or (speed_kmh is not None and speed_kmh >= 25)
        )
        if moving_fast:
            if speed_kmh is not None and speed_kmh >= 70:
                live_score += 0.18
            else:
                live_score += 0.15
            factors.append(_SIGNAL_LABELS["moving_fast"])

        battery_level = _context_float(context, "battery_level")
        battery_charging = _context_bool(context, "battery_charging")
        if battery_level is not None:
            battery_percent = battery_level * 100 if battery_level <= 1 else battery_level
            if battery_percent <= 15 and battery_charging is not True:
                live_score += 0.15
                factors.append(_SIGNAL_LABELS["battery_critical"])
            elif battery_percent <= 25 and battery_charging is not True:
                live_score += 0.05
                factors.append("Battery low")

        network_online = _context_bool(context, "network_online")
        if network_online is False or "network_offline" in requested_signals:
            live_score += 0.15
            factors.append(_SIGNAL_LABELS["network_offline"])

        is_night = _context_bool(context, "is_night")
        if is_night is True or "night_time" in requested_signals:
            live_score += 0.1
            factors.append(_SIGNAL_LABELS["night_time"])

        return list(dict.fromkeys(factors)), _clamp(live_score)

    def analyze(
        self,
        text: str,
        location_present: bool = False,
        context: dict[str, Any] | None = None,
        signals: Iterable[str] | None = None,
    ) -> SafetyScoreResponse:
        normalized_text = " ".join(text.split())
        context = context or {}
        requested_signals = {
            normalized
            for raw_signal in (
                signals if signals is not None else _context_str_list(context, "signals")
            )
            if (normalized := _normalize_signal(raw_signal)) is not None
        }

        live_factors, live_score = self._live_signal_context(
            location_present=location_present,
            context=context,
            requested_signals=requested_signals,
        )
        text_score = self._text_score(normalized_text) if normalized_text else 0.12

        tokens = _tokenize(normalized_text)
        high_matches = sorted(set(tokens) & self._high_tokens)
        medium_matches = sorted(set(tokens) & self._medium_tokens)

        factors = list(live_factors)
        if high_matches:
            factors.append("High-risk language")
        elif medium_matches:
            factors.append("Concerning language")
        elif normalized_text:
            factors.append("No strong risk keywords found")

        if not factors:
            factors.append("Stable conditions")

        threat_score = _clamp(max(text_score, live_score))
        if threat_score >= 0.65:
            threat_level = "high"
        elif threat_score >= 0.35:
            threat_level = "medium"
        else:
            threat_level = "low"

        return SafetyScoreResponse(
            text=normalized_text,
            threat_score=round(threat_score, 2),
            threat_level=threat_level,  # type: ignore[arg-type]
            probabilities=_score_distribution(threat_score),
            factors=factors,
            recommended_actions=list(LABEL_RECOMMENDED_ACTIONS[threat_level]),
        )
