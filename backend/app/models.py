from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class GeoPoint(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)
    accuracy_m: float | None = Field(default=None, ge=0)
    address: str | None = None
    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class UserProfileUpdate(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    avatar_url: str | None = None
    emergency_note: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoredUserProfile(UserProfileUpdate):
    user_id: str
    created_at: datetime
    updated_at: datetime


class ContactCreate(BaseModel):
    user_id: str
    name: str
    phone: str | None = None
    relationship: str | None = None
    address: str | None = None
    avatar_url: str | None = None
    lat: float | None = Field(default=None, ge=-90, le=90)
    lng: float | None = Field(default=None, ge=-180, le=180)
    is_trusted: bool = True
    notes: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StoredContact(ContactCreate):
    id: int
    created_at: datetime
    updated_at: datetime


class ContactListResponse(BaseModel):
    user_id: str
    contacts: list[StoredContact] = Field(default_factory=list)


class NearbyContactsBootstrapRequest(BaseModel):
    user_id: str
    center_lat: float = Field(..., ge=-90, le=90)
    center_lng: float = Field(..., ge=-180, le=180)


class MapLocationCreate(GeoPoint):
    user_id: str


class StoredMapLocation(MapLocationCreate):
    id: int
    created_at: datetime


class MapLocationListResponse(BaseModel):
    user_id: str
    latest: StoredMapLocation | None = None
    locations: list[StoredMapLocation] = Field(default_factory=list)


class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class StoredChatMessage(ChatMessageIn):
    id: int
    user_name: str | None = None
    created_at: datetime


class ChatSafetyAction(BaseModel):
    name: str
    message: str | None = None
    contacts: list[str] = Field(default_factory=list)
    destination_type: Literal["home", "police", "public_place"] | None = None
    timeout_seconds: int | None = Field(default=None, ge=1, le=300)


class ChatRequest(BaseModel):
    user_id: str
    messages: list[ChatMessageIn] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)


class ChatHistoryResponse(BaseModel):
    user_id: str
    messages: list[StoredChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    recommended_actions: list[str] = Field(default_factory=list)
    threat_level: str | None = None
    risk_level: Literal["low", "medium", "high"] = "low"
    confirmation_timeout_seconds: int | None = Field(default=None, ge=1, le=300)
    safety_actions: list[ChatSafetyAction] = Field(default_factory=list)
    history: list[StoredChatMessage] = Field(default_factory=list)


class SafetyScoreRequest(BaseModel):
    user_id: str | None = None
    text: str = ""
    signals: list[str] = Field(default_factory=list)
    location_present: bool = False
    context: dict[str, Any] = Field(default_factory=dict)


class SafetyScoreResponse(BaseModel):
    text: str
    threat_score: float
    threat_level: Literal["low", "medium", "high"]
    probabilities: dict[str, float] = Field(default_factory=dict)
    factors: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)


class AgentRequest(BaseModel):
    user_id: str
    message: str
    mode: Literal["chat", "map", "contacts", "device"] = "chat"
    context: dict[str, Any] = Field(default_factory=dict)
    history: list[ChatMessageIn] = Field(default_factory=list)


class AgentResponse(ChatResponse):
    analysis: SafetyScoreResponse | None = None


class DeviceAlertRequest(BaseModel):
    user_id: str
    device_id: str | None = None
    timestamp: datetime | None = None
    kind: str | None = "panic"
    location: GeoPoint | None = None
    message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DeviceAlertResponse(BaseModel):
    status: str = "ok"
    threat_score: float
    threat_level: Literal["low", "medium", "high"]
    factors: list[str] = Field(default_factory=list)
    event_id: int | None = None
    source: str | None = None


class Esp8266RegisterRequest(BaseModel):
    user_id: str
    device_id: str
    device_label: str | None = None
    firmware_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Esp8266HeartbeatRequest(BaseModel):
    user_id: str
    device_id: str
    battery_voltage: float | None = Field(default=None, ge=0)
    signal_strength: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Esp8266StatusResponse(BaseModel):
    status: str = "ok"
    user_id: str
    device_id: str | None = None
    connected: bool = False
    event_id: int | None = None
    event_type: str | None = None
    last_heartbeat_at: datetime | None = None
    last_alert_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
