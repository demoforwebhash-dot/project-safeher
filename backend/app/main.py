from contextlib import asynccontextmanager
import asyncio
from datetime import datetime, timezone
import json
import logging

from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from .models import (
    AgentRequest,
    AgentResponse,
    ChatHistoryResponse,
    ChatMessageIn,
    ChatRequest,
    ChatResponse,
    ContactListResponse,
    DeviceAlertRequest,
    DeviceAlertResponse,
    Esp8266HeartbeatRequest,
    Esp8266RegisterRequest,
    Esp8266StatusResponse,
    MapLocationCreate,
    MapLocationListResponse,
    NearbyContactsBootstrapRequest,
    SafetyScoreRequest,
    SafetyScoreResponse,
    StoredMapLocation,
    StoredUserProfile,
    UserProfileUpdate,
)
from .state import backend_state


log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    backend_state.attach_esp8266_stream_loop(asyncio.get_running_loop())
    yield


app = FastAPI(title="SafeHer Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = APIRouter(prefix="/v1")


def _format_sse(event: str, data: dict[str, object]) -> str:
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@app.get("/")
def root() -> dict[str, object]:
    return {
        "name": "SafeHer Backend",
        "status": "ok",
        "storage": "neon-postgres",
        "features": ["chat", "contacts", "map", "ml", "agent", "esp8266"],
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "time": datetime.now(timezone.utc).isoformat(),
        "storage": "neon-postgres",
    }


@router.get("/users/{user_id}", response_model=StoredUserProfile)
def get_user_profile(user_id: str) -> StoredUserProfile:
    return backend_state.ensure_user(user_id)


@router.put("/users/{user_id}", response_model=StoredUserProfile)
def update_user_profile(user_id: str, payload: UserProfileUpdate) -> StoredUserProfile:
    return backend_state.upsert_user(user_id, payload)


@router.get("/contacts/{user_id}", response_model=ContactListResponse)
def list_contacts(user_id: str) -> ContactListResponse:
    backend_state.ensure_user(user_id)
    return ContactListResponse(user_id=user_id, contacts=backend_state.list_contacts(user_id))


@router.post("/map/bootstrap-nearby", response_model=ContactListResponse)
def bootstrap_nearby_contacts(
    payload: NearbyContactsBootstrapRequest,
) -> ContactListResponse:
    contacts = backend_state.bootstrap_contacts(
        payload.user_id,
        payload.center_lat,
        payload.center_lng,
    )
    return ContactListResponse(user_id=payload.user_id, contacts=contacts)


@router.get("/map/{user_id}/locations", response_model=MapLocationListResponse)
def list_locations(user_id: str) -> MapLocationListResponse:
    backend_state.ensure_user(user_id)
    locations = backend_state.list_locations(user_id)
    return MapLocationListResponse(
        user_id=user_id,
        latest=locations[0] if locations else None,
        locations=locations,
    )


@router.post("/map/location", response_model=StoredMapLocation, status_code=201)
def save_location(payload: MapLocationCreate) -> StoredMapLocation:
    return backend_state.save_location(payload)


@router.get("/chat/{user_id}", response_model=ChatHistoryResponse)
def get_chat_history(user_id: str) -> ChatHistoryResponse:
    backend_state.ensure_user(user_id)
    return ChatHistoryResponse(user_id=user_id, messages=backend_state.get_chat(user_id))


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not payload.messages:
        raise HTTPException(status_code=400, detail="At least one chat message is required.")

    backend_state.ensure_user(payload.user_id)
    history = backend_state.append_chat(payload.user_id, payload.messages)
    latest_user_message = backend_state.find_latest_user_message(history)
    response = backend_state.build_chat_response(
        user_id=payload.user_id,
        message=latest_user_message,
        context=payload.context,
        history=history,
    )

    backend_state.append_chat(
        payload.user_id,
        [ChatMessageIn(role="assistant", content=response.reply)],
    )

    return response.model_copy(update={"history": backend_state.get_chat(payload.user_id)})


@router.post("/ml/safety-score", response_model=SafetyScoreResponse)
def safety_score(payload: SafetyScoreRequest) -> SafetyScoreResponse:
    has_live_context = payload.location_present or bool(payload.signals) or bool(payload.context)
    if not payload.text.strip() and not has_live_context:
        raise HTTPException(
            status_code=400,
            detail="Text or live signals are required for scoring.",
        )

    return backend_state.score_text(
        payload.text,
        location_present=payload.location_present,
        context=payload.context,
        signals=payload.signals,
    )


@router.websocket("/ml/safety-score/ws")
async def safety_score_ws(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        while True:
            raw_message = await websocket.receive_text()

            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Live prediction payload must be valid JSON.",
                    }
                )
                continue

            if not isinstance(message, dict):
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Live prediction payload must be an object.",
                    }
                )
                continue

            request_id = message.get("request_id")
            if not isinstance(request_id, int):
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Live prediction payload requires an integer request_id.",
                    }
                )
                continue

            try:
                payload = SafetyScoreRequest.model_validate(message)
            except ValidationError as error:
                detail = error.errors()[0].get("msg", "Invalid live prediction payload.")
                await websocket.send_json(
                    {
                        "type": "error",
                        "request_id": request_id,
                        "message": detail,
                    }
                )
                continue

            has_live_context = (
                payload.location_present or bool(payload.signals) or bool(payload.context)
            )
            if not payload.text.strip() and not has_live_context:
                await websocket.send_json(
                    {
                        "type": "error",
                        "request_id": request_id,
                        "message": "Text or live signals are required for scoring.",
                    }
                )
                continue

            analysis = backend_state.score_text(
                payload.text,
                location_present=payload.location_present,
                context=payload.context,
                signals=payload.signals,
            )

            await websocket.send_json(
                {
                    "type": "safety-score-result",
                    "request_id": request_id,
                    "analysis": analysis.model_dump(mode="json"),
                }
            )
    except WebSocketDisconnect:
        return


@router.post("/ai/respond", response_model=AgentResponse)
@router.post("/agent/respond", response_model=AgentResponse)
def agent_respond(payload: AgentRequest) -> AgentResponse:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message is required.")

    backend_state.ensure_user(payload.user_id)
    return backend_state.build_agent_response(
        user_id=payload.user_id,
        message=payload.message,
        mode=payload.mode,
        context=payload.context,
        history=payload.history,
    )


@router.post("/device/alert", response_model=DeviceAlertResponse)
@router.post("/device/esp8266/alert", response_model=DeviceAlertResponse)
def device_alert(payload: DeviceAlertRequest) -> DeviceAlertResponse:
    log.info(
        "Received device alert request user_id=%s device_id=%s kind=%s",
        payload.user_id,
        payload.device_id,
        payload.kind,
    )
    return backend_state.record_device_alert(payload)


@router.get("/device/esp8266/stream/{user_id}")
async def esp8266_alert_stream(
    user_id: str,
    device_id: str | None = None,
) -> StreamingResponse:
    backend_state.ensure_user(user_id)
    queue = backend_state.subscribe_esp8266_alerts(user_id)
    initial_status = backend_state.get_esp8266_status(user_id, device_id)

    async def event_generator():
        try:
            yield ": connected\n\n"
            yield _format_sse("snapshot", initial_status.model_dump(mode="json"))

            while True:
                try:
                    status = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue

                if device_id and status.device_id and status.device_id != device_id:
                    continue

                yield _format_sse("alert", status.model_dump(mode="json"))
        except asyncio.CancelledError:
            return
        finally:
            backend_state.unsubscribe_esp8266_alerts(user_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/device/esp8266/register", response_model=Esp8266StatusResponse)
def esp8266_register(payload: Esp8266RegisterRequest) -> Esp8266StatusResponse:
    backend_state.ensure_user(payload.user_id)
    return backend_state.register_esp8266_device(payload)


@router.post("/device/esp8266/heartbeat", response_model=Esp8266StatusResponse)
def esp8266_heartbeat(payload: Esp8266HeartbeatRequest) -> Esp8266StatusResponse:
    backend_state.ensure_user(payload.user_id)
    return backend_state.record_esp8266_heartbeat(payload)


@router.get("/device/esp8266/status/{user_id}", response_model=Esp8266StatusResponse)
def esp8266_status(user_id: str, device_id: str | None = None) -> Esp8266StatusResponse:
    backend_state.ensure_user(user_id)
    return backend_state.get_esp8266_status(user_id, device_id)


app.include_router(router)
