from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .models import (
    ChatMessageIn,
    ContactCreate,
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


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _database_url() -> str:
    url = os.getenv("NEON_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "Set NEON_DATABASE_URL (or DATABASE_URL) in backend/.env to connect to Neon."
        )
    if "sslmode=" in url:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}sslmode=require"


def _json_dump(value: Any) -> str:
    return json.dumps(value if value is not None else {}, separators=(",", ":"))


def _json_load(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    loaded = json.loads(value)
    return loaded if isinstance(loaded, dict) else {}


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).replace("Z", "+00:00")
    return datetime.fromisoformat(text)


class PostgresStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or _database_url()
        self._pool = ConnectionPool(
            self.database_url,
            kwargs={"row_factory": dict_row},
            open=False,
        )
        self._ready = False
        self._init_lock = threading.RLock()

    def _schema(self) -> str:
        return """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            phone TEXT,
            avatar_url TEXT,
            emergency_note TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS contacts (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            normalized_name TEXT NOT NULL,
            name TEXT NOT NULL,
            phone TEXT,
            relationship TEXT,
            address TEXT,
            avatar_url TEXT,
            lat DOUBLE PRECISION,
            lng DOUBLE PRECISION,
            is_trusted BOOLEAN NOT NULL DEFAULT TRUE,
            notes TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(user_id, normalized_name)
        );

        CREATE TABLE IF NOT EXISTS map_locations (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            lat DOUBLE PRECISION NOT NULL,
            lng DOUBLE PRECISION NOT NULL,
            accuracy_m DOUBLE PRECISION,
            address TEXT,
            label TEXT,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chat_messages (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            user_name TEXT NOT NULL DEFAULT 'SafeHer User',
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS agent_runs (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            input_text TEXT NOT NULL,
            analysis_json TEXT NOT NULL,
            response_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS device_events (
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            device_id TEXT,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            threat_score DOUBLE PRECISION,
            threat_level TEXT,
            message TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS esp8266_devices (
            user_id TEXT NOT NULL,
            device_id TEXT NOT NULL,
            device_label TEXT,
            firmware_version TEXT,
            battery_voltage DOUBLE PRECISION,
            signal_strength INTEGER,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            connected BOOLEAN NOT NULL DEFAULT TRUE,
            last_heartbeat_at TEXT,
            last_alert_at TEXT,
            last_seen_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, device_id)
        );
        """

    def _ensure_ready(self) -> None:
        if self._ready:
            return

        with self._init_lock:
            if self._ready:
                return

            self._pool.open(wait=True)
            with self._pool.connection() as conn:
                with conn.cursor() as cur:
                    for statement in self._schema().split(";"):
                        statement = statement.strip()
                        if statement:
                            cur.execute(statement)
                    cur.execute(
                        """
                        ALTER TABLE chat_messages
                        ADD COLUMN IF NOT EXISTS user_name TEXT
                        """
                    )
                    cur.execute(
                        """
                        UPDATE chat_messages
                        SET user_name = CASE
                            WHEN role = 'assistant' THEN 'safeher-ai'
                            WHEN role = 'system' THEN 'System'
                            ELSE COALESCE(
                                (
                                    SELECT name
                                    FROM users
                                    WHERE users.user_id = chat_messages.user_id
                                ),
                                'SafeHer User'
                            )
                        END
                        WHERE user_name IS NULL OR role IN ('assistant', 'system')
                        """
                    )
                    cur.execute(
                        """
                        ALTER TABLE chat_messages
                        ALTER COLUMN user_name SET DEFAULT 'SafeHer User'
                        """
                    )
                    cur.execute(
                        """
                        ALTER TABLE chat_messages
                        ALTER COLUMN user_name SET NOT NULL
                        """
                    )
                    conn.commit()
            self._ready = True

    @contextmanager
    def _transaction(self) -> Iterator[tuple[Any, Any]]:
        self._ensure_ready()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                try:
                    yield conn, cur
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _avatar_url(seed: str) -> str:
        from urllib.parse import quote

        return f"https://api.dicebear.com/9.x/glass/svg?seed={quote(seed)}"

    @staticmethod
    def _normalize_name(name: str) -> str:
        return " ".join(name.casefold().split())

    @staticmethod
    def _row_or_none(value: Any) -> dict[str, Any] | None:
        return dict(value) if value is not None else None

    def _fetchone(self, query: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        with self._transaction() as (_, cur):
            cur.execute(query, params)
            return self._row_or_none(cur.fetchone())

    def _fetchall(self, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._transaction() as (_, cur):
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    def _user_from_row(self, row: dict[str, Any]) -> StoredUserProfile:
        return StoredUserProfile(
            user_id=row["user_id"],
            name=row["name"],
            email=row["email"],
            phone=row["phone"],
            avatar_url=row["avatar_url"],
            emergency_note=row["emergency_note"],
            metadata=_json_load(row["metadata_json"]),
            created_at=_parse_datetime(row["created_at"]) or datetime.now(timezone.utc),
            updated_at=_parse_datetime(row["updated_at"]) or datetime.now(timezone.utc),
        )

    def _contact_from_row(self, row: dict[str, Any]) -> StoredContact:
        return StoredContact(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            phone=row["phone"],
            relationship=row["relationship"],
            address=row["address"],
            avatar_url=row["avatar_url"],
            lat=row["lat"],
            lng=row["lng"],
            is_trusted=bool(row["is_trusted"]),
            notes=row["notes"],
            metadata=_json_load(row["metadata_json"]),
            created_at=_parse_datetime(row["created_at"]) or datetime.now(timezone.utc),
            updated_at=_parse_datetime(row["updated_at"]) or datetime.now(timezone.utc),
        )

    def _location_from_row(self, row: dict[str, Any]) -> StoredMapLocation:
        return StoredMapLocation(
            id=row["id"],
            user_id=row["user_id"],
            lat=row["lat"],
            lng=row["lng"],
            accuracy_m=row["accuracy_m"],
            address=row["address"],
            label=row["label"],
            metadata=_json_load(row["metadata_json"]),
            created_at=_parse_datetime(row["created_at"]) or datetime.now(timezone.utc),
        )

    def _chat_from_row(self, row: dict[str, Any]) -> StoredChatMessage:
        return StoredChatMessage(
            id=row["id"],
            role=row["role"],
            content=row["content"],
            user_name=row.get("user_name"),
            created_at=_parse_datetime(row["created_at"]) or datetime.now(timezone.utc),
        )

    def _device_from_row(self, row: dict[str, Any]) -> Esp8266StatusResponse:
        return Esp8266StatusResponse(
            status="ok",
            user_id=row["user_id"],
            device_id=row["device_id"],
            connected=bool(row["connected"]),
            last_heartbeat_at=_parse_datetime(row["last_heartbeat_at"]),
            last_alert_at=_parse_datetime(row["last_alert_at"]),
            metadata=_json_load(row["metadata_json"]),
        )

    def ensure_user(self, user_id: str) -> StoredUserProfile:
        row = self._fetchone("SELECT * FROM users WHERE user_id = %s", (user_id,))
        if row is not None:
            return self._user_from_row(row)

        now = self._now_iso()
        with self._transaction() as (_, cur):
            cur.execute(
                """
                INSERT INTO users (
                    user_id, name, email, phone, avatar_url, emergency_note,
                    metadata_json, created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id) DO NOTHING
                """,
                (
                    user_id,
                    "SafeHer User",
                    None,
                    None,
                    self._avatar_url(user_id),
                    "Default profile generated by the backend.",
                    _json_dump({"source": "neon"}),
                    now,
                    now,
                ),
            )

        row = self._fetchone("SELECT * FROM users WHERE user_id = %s", (user_id,))
        if row is None:
            raise RuntimeError("Unable to create user profile.")
        return self._user_from_row(row)

    def upsert_user(self, user_id: str, payload: UserProfileUpdate) -> StoredUserProfile:
        existing = self.ensure_user(user_id)
        merged_metadata = dict(existing.metadata)
        merged_metadata.update(payload.metadata or {})
        now = self._now_iso()

        with self._transaction() as (_, cur):
            cur.execute(
                """
                UPDATE users
                SET
                    name = %s,
                    email = %s,
                    phone = %s,
                    avatar_url = %s,
                    emergency_note = %s,
                    metadata_json = %s,
                    updated_at = %s
                WHERE user_id = %s
                """,
                (
                    payload.name if payload.name is not None else existing.name,
                    payload.email if payload.email is not None else existing.email,
                    payload.phone if payload.phone is not None else existing.phone,
                    payload.avatar_url if payload.avatar_url is not None else existing.avatar_url,
                    payload.emergency_note
                    if payload.emergency_note is not None
                    else existing.emergency_note,
                    _json_dump(merged_metadata),
                    now,
                    user_id,
                ),
            )

        row = self._fetchone("SELECT * FROM users WHERE user_id = %s", (user_id,))
        if row is None:
            raise RuntimeError("Unable to update user profile.")
        return self._user_from_row(row)

    def upsert_contact(self, payload: ContactCreate) -> StoredContact:
        self.ensure_user(payload.user_id)
        now = self._now_iso()
        row = self._fetchone(
            """
            INSERT INTO contacts (
                user_id, normalized_name, name, phone, relationship, address,
                avatar_url, lat, lng, is_trusted, notes, metadata_json,
                created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, normalized_name)
            DO UPDATE SET
                name = EXCLUDED.name,
                phone = EXCLUDED.phone,
                relationship = EXCLUDED.relationship,
                address = EXCLUDED.address,
                avatar_url = EXCLUDED.avatar_url,
                lat = EXCLUDED.lat,
                lng = EXCLUDED.lng,
                is_trusted = EXCLUDED.is_trusted,
                notes = EXCLUDED.notes,
                metadata_json = EXCLUDED.metadata_json,
                updated_at = EXCLUDED.updated_at
            RETURNING *
            """,
            (
                payload.user_id,
                self._normalize_name(payload.name),
                payload.name,
                payload.phone,
                payload.relationship,
                payload.address,
                payload.avatar_url,
                payload.lat,
                payload.lng,
                payload.is_trusted,
                payload.notes,
                _json_dump(payload.metadata or {}),
                now,
                now,
            ),
        )
        if row is None:
            raise RuntimeError("Unable to save contact.")
        return self._contact_from_row(row)

    def list_contacts(self, user_id: str) -> list[StoredContact]:
        return [self._contact_from_row(row) for row in self._fetchall(
            """
            SELECT * FROM contacts
            WHERE user_id = %s
            ORDER BY is_trusted DESC, id ASC
            """,
            (user_id,),
        )]

    def save_location(self, payload: MapLocationCreate) -> StoredMapLocation:
        self.ensure_user(payload.user_id)
        row = self._fetchone(
            """
            INSERT INTO map_locations (
                user_id, lat, lng, accuracy_m, address, label,
                metadata_json, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING *
            """,
            (
                payload.user_id,
                payload.lat,
                payload.lng,
                payload.accuracy_m,
                payload.address,
                payload.label,
                _json_dump(payload.metadata or {}),
                self._now_iso(),
            ),
        )
        if row is None:
            raise RuntimeError("Unable to save location.")
        return self._location_from_row(row)

    def list_locations(self, user_id: str) -> list[StoredMapLocation]:
        return [self._location_from_row(row) for row in self._fetchall(
            """
            SELECT * FROM map_locations
            WHERE user_id = %s
            ORDER BY id DESC
            """,
            (user_id,),
        )]

    def append_chat(self, user_id: str, messages: list[ChatMessageIn]) -> list[StoredChatMessage]:
        if not messages:
            return self.get_chat(user_id)

        user_profile = self.ensure_user(user_id)
        user_name = (user_profile.name or "SafeHer User").strip() or "SafeHer User"
        history = self.get_chat(user_id)
        existing_signature = [(item.role, item.content) for item in history]
        incoming_signature = [(item.role, item.content) for item in messages]

        if (
            len(incoming_signature) >= len(existing_signature)
            and incoming_signature[: len(existing_signature)] == existing_signature
        ):
            messages = messages[len(existing_signature) :]

        if not messages:
            return history

        with self._transaction() as (_, cur):
            for message in messages:
                display_name = {
                    "assistant": "safeher-ai",
                    "system": "System",
                }.get(message.role, user_name)
                cur.execute(
                    """
                    INSERT INTO chat_messages (user_id, user_name, role, content, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (
                        user_id,
                        display_name,
                        message.role,
                        message.content,
                        self._now_iso(),
                    ),
                )

        return self.get_chat(user_id)

    def get_chat(self, user_id: str) -> list[StoredChatMessage]:
        return [self._chat_from_row(row) for row in self._fetchall(
            """
            SELECT * FROM chat_messages
            WHERE user_id = %s
            ORDER BY id ASC
            """,
            (user_id,),
        )]

    def record_agent_run(
        self,
        *,
        user_id: str,
        input_text: str,
        analysis: SafetyScoreResponse,
        response_text: str,
    ) -> int:
        row = self._fetchone(
            """
            INSERT INTO agent_runs (
                user_id, input_text, analysis_json, response_text, created_at
            )
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                user_id,
                input_text,
                _json_dump(analysis.model_dump(mode="json")),
                response_text,
                self._now_iso(),
            ),
        )
        if row is None:
            raise RuntimeError("Unable to record agent run.")
        return int(row["id"])

    def record_device_event(
        self,
        *,
        user_id: str,
        device_id: str | None,
        event_type: str,
        payload: dict[str, Any],
        threat_score: float | None = None,
        threat_level: str | None = None,
        message: str | None = None,
    ) -> int:
        row = self._fetchone(
            """
            INSERT INTO device_events (
                user_id, device_id, event_type, payload_json,
                threat_score, threat_level, message, created_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                user_id,
                device_id,
                event_type,
                _json_dump(payload),
                threat_score,
                threat_level,
                message,
                self._now_iso(),
            ),
        )
        if row is None:
            raise RuntimeError("Unable to record device event.")
        return int(row["id"])

    def _upsert_esp8266_device(
        self,
        *,
        user_id: str,
        device_id: str,
        device_label: str | None = None,
        firmware_version: str | None = None,
        battery_voltage: float | None = None,
        signal_strength: int | None = None,
        metadata: dict[str, Any] | None = None,
        connected: bool = True,
        last_heartbeat_at: str | None = None,
        last_alert_at: str | None = None,
    ) -> Esp8266StatusResponse:
        existing = self._fetchone(
            """
            SELECT * FROM esp8266_devices
            WHERE user_id = %s AND device_id = %s
            """,
            (user_id, device_id),
        )
        merged_metadata = dict(_json_load(existing["metadata_json"])) if existing else {}
        merged_metadata.update(metadata or {})
        now = self._now_iso()

        row = self._fetchone(
            """
            INSERT INTO esp8266_devices (
                user_id, device_id, device_label, firmware_version,
                battery_voltage, signal_strength, metadata_json, connected,
                last_heartbeat_at, last_alert_at, last_seen_at, created_at, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, device_id)
            DO UPDATE SET
                device_label = EXCLUDED.device_label,
                firmware_version = EXCLUDED.firmware_version,
                battery_voltage = EXCLUDED.battery_voltage,
                signal_strength = EXCLUDED.signal_strength,
                metadata_json = EXCLUDED.metadata_json,
                connected = EXCLUDED.connected,
                last_heartbeat_at = COALESCE(EXCLUDED.last_heartbeat_at, esp8266_devices.last_heartbeat_at),
                last_alert_at = COALESCE(EXCLUDED.last_alert_at, esp8266_devices.last_alert_at),
                last_seen_at = EXCLUDED.last_seen_at,
                updated_at = EXCLUDED.updated_at
            RETURNING *
            """,
            (
                user_id,
                device_id,
                device_label if device_label is not None else (existing["device_label"] if existing else None),
                firmware_version if firmware_version is not None else (existing["firmware_version"] if existing else None),
                battery_voltage if battery_voltage is not None else (existing["battery_voltage"] if existing else None),
                signal_strength if signal_strength is not None else (existing["signal_strength"] if existing else None),
                _json_dump(merged_metadata),
                connected,
                last_heartbeat_at,
                last_alert_at,
                now,
                now,
                now,
            ),
        )
        if row is None:
            raise RuntimeError("Unable to save ESP8266 device.")
        return self._device_from_row(row)

    def register_esp8266_device(self, payload: Esp8266RegisterRequest) -> Esp8266StatusResponse:
        return self._upsert_esp8266_device(
            user_id=payload.user_id,
            device_id=payload.device_id,
            device_label=payload.device_label,
            firmware_version=payload.firmware_version,
            metadata=payload.metadata,
            connected=True,
            last_heartbeat_at=self._now_iso(),
        )

    def record_esp8266_heartbeat(self, payload: Esp8266HeartbeatRequest) -> Esp8266StatusResponse:
        return self._upsert_esp8266_device(
            user_id=payload.user_id,
            device_id=payload.device_id,
            battery_voltage=payload.battery_voltage,
            signal_strength=payload.signal_strength,
            metadata=payload.metadata,
            connected=True,
            last_heartbeat_at=self._now_iso(),
        )

    def record_esp8266_alert(
        self,
        *,
        user_id: str,
        device_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> Esp8266StatusResponse:
        return self._upsert_esp8266_device(
            user_id=user_id,
            device_id=device_id,
            metadata=metadata,
            connected=True,
            last_alert_at=self._now_iso(),
        )

    def get_esp8266_status(self, user_id: str, device_id: str | None) -> Esp8266StatusResponse:
        if not device_id:
            row = self._fetchone(
                """
                SELECT * FROM esp8266_devices
                WHERE user_id = %s
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """,
                (user_id,),
            )
        else:
            row = self._fetchone(
                """
                SELECT * FROM esp8266_devices
                WHERE user_id = %s AND device_id = %s
                """,
                (user_id, device_id),
            )

        if row is None:
            return Esp8266StatusResponse(user_id=user_id, device_id=device_id, connected=False)

        return self._device_from_row(row)


DatabaseStore = PostgresStore
