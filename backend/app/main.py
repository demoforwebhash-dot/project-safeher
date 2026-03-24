from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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


@asynccontextmanager
async def lifespan(_: FastAPI):
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
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text is required for scoring.")

    return backend_state.score_text(
        payload.text,
        location_present=payload.location_present,
        context=payload.context,
    )


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
    return backend_state.record_device_alert(payload)


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
