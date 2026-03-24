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


def _softmax(log_values: dict[str, float]) -> dict[str, float]:
    if not log_values:
        return {label: 0.0 for label in LABELS}

    max_value = max(log_values.values())
    exp_values = {label: math.exp(value - max_value) for label, value in log_values.items()}
    total = sum(exp_values.values()) or 1.0
    return {label: value / total for label, value in exp_values.items()}


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

    def analyze(
        self,
        text: str,
        location_present: bool = False,
        context: dict[str, Any] | None = None,
    ) -> SafetyScoreResponse:
        normalized_text = " ".join(text.split())
        context = context or {}
        probabilities = self._predict_probabilities(normalized_text)
        base_score = sum(probabilities[label] * LABEL_SCORES[label] for label in LABELS)
        factors: list[str] = []
        tokens = _tokenize(normalized_text)
        context_adjustment = 0.0

        high_matches = sorted(set(tokens) & self._high_tokens)
        medium_matches = sorted(set(tokens) & self._medium_tokens)

        if high_matches:
            factors.append(
                f"High-risk language matched: {', '.join(high_matches[:4])}."
            )
        elif medium_matches:
            factors.append(
                f"Potential concern language matched: {', '.join(medium_matches[:4])}."
            )
        elif normalized_text:
            factors.append("No strong risk keywords were found.")
        else:
            factors.append("No message content was provided.")

        location_status = (_context_str(context, "location_status") or "").casefold()
        if location_present or location_status in {"watching", "available", "live"}:
            context_adjustment += 0.03
            factors.append("A live location is available for support.")
        elif location_status in {"denied", "unavailable", "unsupported"}:
            context_adjustment += 0.14
            factors.append("Location signal is unavailable.")

        gps_accuracy_m = _context_float(context, "gps_accuracy_m")
        if gps_accuracy_m is not None:
            if gps_accuracy_m > 150:
                context_adjustment += 0.08
                factors.append(f"GPS accuracy is poor at {gps_accuracy_m:.0f}m.")
            elif gps_accuracy_m > 50:
                context_adjustment += 0.04
                factors.append(f"GPS accuracy is moderate at {gps_accuracy_m:.0f}m.")
            else:
                factors.append(f"GPS accuracy is strong at {gps_accuracy_m:.0f}m.")

        speed_kmh = _context_float(context, "speed_kmh")
        if speed_kmh is not None:
            if speed_kmh >= 70:
                context_adjustment += 0.22
                factors.append(f"Rapid movement detected at {speed_kmh:.0f} km/h.")
            elif speed_kmh >= 25:
                context_adjustment += 0.12
                factors.append(f"Fast movement detected at {speed_kmh:.0f} km/h.")
            elif speed_kmh >= 5:
                context_adjustment += 0.06
                factors.append(f"Movement detected at {speed_kmh:.0f} km/h.")
            else:
                factors.append("User appears stationary.")

        movement_state = _context_str(context, "movement_state")
        if movement_state in {"fast", "moving"}:
            context_adjustment += 0.05
            factors.append(f"Movement state reports {movement_state}.")

        battery_level = _context_float(context, "battery_level")
        battery_charging = _context_bool(context, "battery_charging")
        if battery_level is not None:
            battery_percent = battery_level * 100 if battery_level <= 1 else battery_level
            if battery_percent <= 10 and not battery_charging:
                context_adjustment += 0.08
                factors.append(f"Battery is critically low at {battery_percent:.0f}%.")
            elif battery_percent <= 20 and not battery_charging:
                context_adjustment += 0.04
                factors.append(f"Battery is getting low at {battery_percent:.0f}%.")
            else:
                factors.append(f"Battery level is {battery_percent:.0f}%.")

        network_online = _context_bool(context, "network_online")
        if network_online is False:
            context_adjustment += 0.04
            factors.append("Network connectivity is offline.")

        is_night = _context_bool(context, "is_night")
        if is_night is True:
            context_adjustment += 0.06
            factors.append("Night-time conditions are active.")

        threat_score = _clamp(base_score + context_adjustment)
        if threat_score >= 0.7:
            threat_level = "high"
        elif threat_score >= 0.4:
            threat_level = "medium"
        else:
            threat_level = "low"

        if threat_level == "high" and not high_matches and context_adjustment > 0:
            factors.append("The trained model still classified the message as high risk.")

        return SafetyScoreResponse(
            text=normalized_text,
            threat_score=round(threat_score, 2),
            threat_level=threat_level,  # type: ignore[arg-type]
            probabilities=probabilities,
            factors=factors,
            recommended_actions=list(LABEL_RECOMMENDED_ACTIONS[threat_level]),
        )
