from __future__ import annotations

import asyncio
from collections import defaultdict
import logging
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Iterable

from .ai import GroqReplyGenerator
from .db import PostgresStore
from .ml import SafetyRiskModel
from .models import (
    AgentResponse,
    ChatMessageIn,
    ChatResponse,
    ChatSafetyAction,
    ContactCreate,
    DeviceAlertRequest,
    DeviceAlertResponse,
    Esp8266HeartbeatRequest,
    Esp8266RegisterRequest,
    Esp8266StatusResponse,
    MapLocationCreate,
    SafetyScoreResponse,
    StoredChatMessage,
    StoredContact,
    StoredMapLocation,
    StoredUserProfile,
    UserProfileUpdate,
)


def _avatar_url(seed: str) -> str:
    from urllib.parse import quote

    return f"https://api.dicebear.com/9.x/glass/svg?seed={quote(seed)}"


def _compass_label(bearing_deg: float) -> str:
    directions = (
        "north",
        "north-east",
        "east",
        "south-east",
        "south",
        "south-west",
        "west",
        "north-west",
    )
    index = int((bearing_deg + 22.5) // 45) % len(directions)
    return directions[index]


def _destination_point(
    lat: float,
    lng: float,
    distance_m: float,
    bearing_deg: float,
) -> tuple[float, float]:
    earth_radius_m = 6_371_000
    angular_distance = distance_m / earth_radius_m
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lng1 = math.radians(lng)

    lat2 = math.asin(
        math.sin(lat1) * math.cos(angular_distance)
        + math.cos(lat1) * math.sin(angular_distance) * math.cos(bearing)
    )
    lng2 = lng1 + math.atan2(
        math.sin(bearing) * math.sin(angular_distance) * math.cos(lat1),
        math.cos(angular_distance) - math.sin(lat1) * math.sin(lat2),
    )
    return math.degrees(lat2), ((math.degrees(lng2) + 540) % 360) - 180


@dataclass(frozen=True)
class ContactSeed:
    name: str
    relationship: str
    phone: str
    bearing_deg: float
    avatar_seed: str
    notes: str

    @property
    def direction_label(self) -> str:
        return _compass_label(self.bearing_deg)


def _contact_distance_m() -> float:
    try:
        return float(os.getenv("SAFEHER_CONTACT_DISTANCE_M", "500"))
    except ValueError:
        return 500.0


DEFAULT_CONTACT_DISTANCE_M = _contact_distance_m()

DEFAULT_CONTACT_SEEDS: tuple[ContactSeed, ...] = (
    ContactSeed(
        name="Father",
        relationship="father",
        phone="+1-555-0101",
        bearing_deg=315.0,
        avatar_seed="father",
        notes="Seeded emergency contact for the user's father.",
    ),
    ContactSeed(
        name="Mother",
        relationship="mother",
        phone="+1-555-0102",
        bearing_deg=45.0,
        avatar_seed="mother",
        notes="Seeded emergency contact for the user's mother.",
    ),
    ContactSeed(
        name="Police",
        relationship="",
        phone="+1-555-0103",
        bearing_deg=225.0,
        avatar_seed="police",
        notes="Seeded emergency contact for local police support.",
    ),
)

log = logging.getLogger(__name__)


class Esp8266AlertStreamHub:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: dict[str, set[asyncio.Queue[Esp8266StatusResponse]]] = defaultdict(set)
        self._lock = threading.RLock()

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._loop = loop

    def subscribe(self, user_id: str) -> asyncio.Queue[Esp8266StatusResponse]:
        queue: asyncio.Queue[Esp8266StatusResponse] = asyncio.Queue()
        with self._lock:
            self._subscribers[user_id].add(queue)
        return queue

    def unsubscribe(self, user_id: str, queue: asyncio.Queue[Esp8266StatusResponse]) -> None:
        with self._lock:
            subscribers = self._subscribers.get(user_id)
            if not subscribers:
                return

            subscribers.discard(queue)
            if not subscribers:
                self._subscribers.pop(user_id, None)

    def publish(self, user_id: str, status: Esp8266StatusResponse) -> None:
        with self._lock:
            loop = self._loop
            subscribers = tuple(self._subscribers.get(user_id, ()))

        if loop is None or not subscribers:
            return

        for queue in subscribers:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, status)
            except RuntimeError:
                continue


class SafetyAgent:
    def __init__(self) -> None:
        self._reply_generator = GroqReplyGenerator()

    def _fallback_reply(self, threat_level: str) -> str:
        if threat_level == "high":
            return (
                "I'm taking this seriously. Move toward a public place if you can, "
                "share your live location, and call emergency services if you are in immediate danger."
            )

        if threat_level == "medium":
            return (
                "That sounds concerning. Stay near other people, keep your phone charged, "
                "and share your location with someone you trust."
            )

        return (
            "I'm here if you need anything. I can help with safety planning, nearby contacts, "
            "or live location sharing."
        )

    def _generate_reply(
        self,
        *,
        mode: str,
        analysis: SafetyScoreResponse,
        contacts: list[str],
        message: str,
        history: list[StoredChatMessage] | None = None,
        context: dict[str, Any] | None = None,
    ) -> str:
        if self._reply_generator.available:
            try:
                return self._reply_generator.generate_reply(
                    mode=mode,
                    analysis=analysis,
                    contacts=contacts,
                    message=message,
                    history=history,
                    context=context,
                )
            except Exception:
                log.exception("Groq reply generation failed; falling back to local reply.")

        return self._fallback_reply(analysis.threat_level)

    def _build_action_plan(
        self,
        analysis: SafetyScoreResponse,
        contacts: list[str],
    ) -> tuple[list[str], list[ChatSafetyAction], int | None]:
        emergency_contacts = contacts[:3] if contacts else ["Father", "Mother", "Police"]

        if analysis.threat_level == "high":
            recommended_actions = [
                "trigger_sos",
                "share_live_location",
                "call_emergency_contact",
                "notify_authorities",
            ]
            safety_actions = [
                ChatSafetyAction(
                    name="trigger_sos",
                    message="Trigger an SOS alert now.",
                    timeout_seconds=15,
                ),
                ChatSafetyAction(
                    name="share_live_location",
                    message="Share your live location with trusted contacts.",
                    contacts=emergency_contacts,
                    timeout_seconds=30,
                ),
                ChatSafetyAction(
                    name="call_emergency_contact",
                    message="Call your emergency contact now.",
                    contacts=emergency_contacts[:2],
                    timeout_seconds=30,
                ),
                ChatSafetyAction(
                    name="notify_authorities",
                    message="Contact local authorities if you are in immediate danger.",
                    destination_type="police",
                    timeout_seconds=30,
                ),
            ]
            return recommended_actions, safety_actions, 20

        if analysis.threat_level == "medium":
            recommended_actions = [
                "share_live_location",
                "call_emergency_contact",
                "find_safe_route",
            ]
            safety_actions = [
                ChatSafetyAction(
                    name="share_live_location",
                    message="Share your live location with trusted contacts.",
                    contacts=emergency_contacts,
                    timeout_seconds=30,
                ),
                ChatSafetyAction(
                    name="call_emergency_contact",
                    message="Call your emergency contact.",
                    contacts=emergency_contacts[:2],
                    timeout_seconds=30,
                ),
                ChatSafetyAction(
                    name="find_safe_route",
                    message="Find a safe route to a public place or trusted contact.",
                    destination_type="public_place",
                    timeout_seconds=60,
                ),
            ]
            return recommended_actions, safety_actions, 30

        return [], [], None

    def build_chat_response(
        self,
        *,
        analysis: SafetyScoreResponse,
        contacts: list[str],
        message: str,
        history: list[StoredChatMessage] | None = None,
    ) -> ChatResponse:
        reply = self._generate_reply(
            mode="chat",
            analysis=analysis,
            contacts=contacts,
            message=message,
            history=history,
        )
        recommended_actions, safety_actions, confirmation_timeout_seconds = self._build_action_plan(
            analysis,
            contacts,
        )

        return ChatResponse(
            reply=reply,
            recommended_actions=recommended_actions,
            threat_level=analysis.threat_level,
            risk_level=analysis.threat_level,
            confirmation_timeout_seconds=confirmation_timeout_seconds,
            safety_actions=safety_actions,
            history=history or [],
        )

    def build_agent_response(
        self,
        *,
        analysis: SafetyScoreResponse,
        contacts: list[str],
        message: str,
        history: list[StoredChatMessage] | None = None,
        mode: str = "chat",
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        reply = self._generate_reply(
            mode=mode,
            analysis=analysis,
            contacts=contacts,
            message=message,
            history=history,
            context=context,
        )
        recommended_actions, safety_actions, confirmation_timeout_seconds = self._build_action_plan(
            analysis,
            contacts,
        )

        return AgentResponse(
            reply=reply,
            recommended_actions=recommended_actions,
            threat_level=analysis.threat_level,
            risk_level=analysis.threat_level,
            confirmation_timeout_seconds=confirmation_timeout_seconds,
            safety_actions=safety_actions,
            history=history or [],
            analysis=analysis,
        )


class Esp8266Gateway:
    def __init__(self, store: PostgresStore) -> None:
        self._store = store

    def register_device(self, payload: Esp8266RegisterRequest) -> Esp8266StatusResponse:
        status = self._store.register_esp8266_device(payload)
        event_id = self._store.record_device_event(
            user_id=payload.user_id,
            device_id=payload.device_id,
            event_type="register",
            payload={
                "device_label": payload.device_label,
                "firmware_version": payload.firmware_version,
                "metadata": payload.metadata,
            },
            message="ESP8266 device registered.",
        )
        return status.model_copy(update={"event_id": event_id, "event_type": "register"})

    def record_heartbeat(self, payload: Esp8266HeartbeatRequest) -> Esp8266StatusResponse:
        status = self._store.record_esp8266_heartbeat(payload)
        event_id = self._store.record_device_event(
            user_id=payload.user_id,
            device_id=payload.device_id,
            event_type="heartbeat",
            payload={
                "battery_voltage": payload.battery_voltage,
                "signal_strength": payload.signal_strength,
                "metadata": payload.metadata,
            },
            message="ESP8266 heartbeat received.",
        )
        return status.model_copy(update={"event_id": event_id, "event_type": "heartbeat"})

    def get_status(
        self,
        *,
        user_id: str,
        device_id: str | None,
    ) -> Esp8266StatusResponse:
        return self._store.get_esp8266_status(user_id, device_id)


class SafeHerBackend:
    def __init__(
        self,
        store: PostgresStore | None = None,
        model: SafetyRiskModel | None = None,
    ) -> None:
        self.store = store or PostgresStore()
        self.model = model or SafetyRiskModel()
        self.agent = SafetyAgent()
        self.esp8266 = Esp8266Gateway(self.store)
        self._esp8266_alert_streams = Esp8266AlertStreamHub()

    def attach_esp8266_stream_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._esp8266_alert_streams.attach_loop(loop)

    def ensure_user(self, user_id: str) -> StoredUserProfile:
        return self.store.ensure_user(user_id)

    def upsert_user(self, user_id: str, payload: UserProfileUpdate) -> StoredUserProfile:
        return self.store.upsert_user(user_id, payload)

    def list_contacts(self, user_id: str) -> list[StoredContact]:
        return self.store.list_contacts(user_id)

    def trusted_contact_names(self, user_id: str, limit: int = 3) -> list[str]:
        contacts = [contact.name for contact in self.list_contacts(user_id) if contact.is_trusted]
        if contacts:
            return contacts[:limit]
        return ["Father", "Mother", "Police"][:limit]

    def bootstrap_contacts(
        self,
        user_id: str,
        center_lat: float,
        center_lng: float,
    ) -> list[StoredContact]:
        self.ensure_user(user_id)

        for seed in DEFAULT_CONTACT_SEEDS:
            contact_lat, contact_lng = _destination_point(
                center_lat,
                center_lng,
                DEFAULT_CONTACT_DISTANCE_M,
                seed.bearing_deg,
            )
            self.store.upsert_contact(
                ContactCreate(
                    user_id=user_id,
                    name=seed.name,
                    phone=seed.phone,
                    relationship=seed.relationship,
                    address=(
                        f"About {DEFAULT_CONTACT_DISTANCE_M:.0f}m {seed.direction_label} of your current location "
                        f"({contact_lat:.4f}, {contact_lng:.4f})"
                    ),
                    avatar_url=_avatar_url(seed.avatar_seed),
                    lat=contact_lat,
                    lng=contact_lng,
                    is_trusted=True,
                    notes=seed.notes,
                    metadata={
                        "seeded_contact": True,
                        "seeded_from_live_location": True,
                        "seeded_distance_m": DEFAULT_CONTACT_DISTANCE_M,
                        "seeded_bearing_deg": seed.bearing_deg,
                        "seed_anchor_lat": center_lat,
                        "seed_anchor_lng": center_lng,
                    },
                )
            )

        return self.list_contacts(user_id)

    def list_locations(self, user_id: str) -> list[StoredMapLocation]:
        return self.store.list_locations(user_id)

    def save_location(self, payload: MapLocationCreate) -> StoredMapLocation:
        return self.store.save_location(payload)

    def get_chat(self, user_id: str) -> list[StoredChatMessage]:
        return self.store.get_chat(user_id)

    def append_chat(self, user_id: str, messages: list[ChatMessageIn]) -> list[StoredChatMessage]:
        return self.store.append_chat(user_id, messages)

    @staticmethod
    def find_latest_user_message(messages: list[ChatMessageIn]) -> str:
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        return messages[-1].content if messages else ""

    def score_text(
        self,
        text: str,
        location_present: bool = False,
        context: dict[str, Any] | None = None,
        signals: Iterable[str] | None = None,
    ) -> SafetyScoreResponse:
        return self.model.analyze(
            text,
            location_present=location_present,
            context=context,
            signals=signals,
        )

    def build_chat_response(
        self,
        *,
        user_id: str,
        message: str,
        context: dict[str, Any] | None = None,
        history: list[StoredChatMessage] | None = None,
    ) -> ChatResponse:
        analysis = self.score_text(
            message,
            location_present=bool((context or {}).get("location_present")),
            context=context,
        )
        response = self.agent.build_chat_response(
            analysis=analysis,
            contacts=self.trusted_contact_names(user_id),
            message=message,
            history=history,
        )
        self.store.record_agent_run(
            user_id=user_id,
            input_text=message,
            analysis=analysis,
            response_text=response.reply,
        )
        return response

    def build_agent_response(
        self,
        *,
        user_id: str,
        message: str,
        context: dict[str, Any] | None = None,
        history: list[ChatMessageIn] | None = None,
        mode: str = "chat",
    ) -> AgentResponse:
        analysis = self.score_text(
            message,
            location_present=bool((context or {}).get("location_present")),
            context=context,
        )
        response = self.agent.build_agent_response(
            analysis=analysis,
            contacts=self.trusted_contact_names(user_id),
            message=message,
            history=history,
            mode=mode,
            context=context,
        )
        self.store.record_agent_run(
            user_id=user_id,
            input_text=message,
            analysis=analysis,
            response_text=response.reply,
        )
        return response.model_copy(update={"history": []})

    def record_agent_run(
        self,
        *,
        user_id: str,
        input_text: str,
        analysis: SafetyScoreResponse,
        response_text: str,
    ) -> int:
        return self.store.record_agent_run(
            user_id=user_id,
            input_text=input_text,
            analysis=analysis,
            response_text=response_text,
        )

    def record_device_alert(self, payload: DeviceAlertRequest) -> DeviceAlertResponse:
        self.ensure_user(payload.user_id)
        log.info(
            "Processing device alert user_id=%s device_id=%s kind=%s",
            payload.user_id,
            payload.device_id,
            payload.kind,
        )
        signal_text = " ".join(
            part for part in (payload.kind or "panic", payload.message or "") if part
        )
        analysis = self.score_text(
            signal_text,
            location_present=payload.location is not None,
            context={
                "location_status": "available" if payload.location is not None else "unavailable",
                "gps_accuracy_m": payload.location.accuracy_m if payload.location is not None else None,
                "movement_state": "stationary",
                "network_online": True,
                "is_night": False,
            },
        )

        if payload.location is not None:
            self.save_location(
                MapLocationCreate(
                    user_id=payload.user_id,
                    lat=payload.location.lat,
                    lng=payload.location.lng,
                    accuracy_m=payload.location.accuracy_m,
                    address=payload.location.address,
                    label=payload.kind or "device-alert",
                    metadata={
                        "source": "device_alert",
                        "device_id": payload.device_id,
                        "message": payload.message,
                    },
                )
            )

        event_id = self.store.record_device_event(
            user_id=payload.user_id,
            device_id=payload.device_id,
            event_type="alert",
            payload=payload.model_dump(mode="json"),
            threat_score=analysis.threat_score,
            threat_level=analysis.threat_level,
            message=payload.message or payload.kind or "alert",
        )

        if payload.device_id:
            device_status = self.store.record_esp8266_alert(
                user_id=payload.user_id,
                device_id=payload.device_id,
                metadata={
                    "kind": payload.kind,
                    "message": payload.message,
                    "alert_kind": payload.kind,
                    "alert_message": payload.message,
                    "alert_threat_score": analysis.threat_score,
                    "alert_threat_level": analysis.threat_level,
                    "alert_factors": analysis.factors,
                },
            ).model_copy(update={"event_id": event_id, "event_type": "alert"})
            self.publish_esp8266_alert(payload.user_id, device_status)

        log.info(
            "Recorded device alert event_id=%s threat_level=%s source=%s",
            event_id,
            analysis.threat_level,
            payload.kind or "panic",
        )

        return DeviceAlertResponse(
            status="ok",
            threat_score=analysis.threat_score,
            threat_level=analysis.threat_level,
            factors=analysis.factors,
            event_id=event_id,
            source=payload.kind or "panic",
        )

    def register_esp8266_device(self, payload: Esp8266RegisterRequest) -> Esp8266StatusResponse:
        return self.esp8266.register_device(payload)

    def record_esp8266_heartbeat(self, payload: Esp8266HeartbeatRequest) -> Esp8266StatusResponse:
        return self.esp8266.record_heartbeat(payload)

    def get_esp8266_status(self, user_id: str, device_id: str | None) -> Esp8266StatusResponse:
        return self.esp8266.get_status(user_id=user_id, device_id=device_id)

    def subscribe_esp8266_alerts(self, user_id: str) -> asyncio.Queue[Esp8266StatusResponse]:
        return self._esp8266_alert_streams.subscribe(user_id)

    def unsubscribe_esp8266_alerts(
        self,
        user_id: str,
        queue: asyncio.Queue[Esp8266StatusResponse],
    ) -> None:
        self._esp8266_alert_streams.unsubscribe(user_id, queue)

    def publish_esp8266_alert(self, user_id: str, status: Esp8266StatusResponse) -> None:
        self._esp8266_alert_streams.publish(user_id, status)

