# SafeHer Backend

## Quickstart

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API base: `http://localhost:8000`

This backend now uses Neon/Postgres for persistence. Set `NEON_DATABASE_URL` in `backend/.env` with your Neon connection string. The app will also accept `DATABASE_URL` as a fallback.
If you want the chat and agent replies to come from LangChain + Groq, also set `GROQ_API_KEY`. The backend uses LangChain's `ChatGroq` integration and falls back to the local safety reply generator when the key is missing.

## What It Covers

- Chat history persistence for the frontend chat page
- Emergency contacts seeded as `Father`, `Mother`, and `Police`
- Live map locations for the current user
- A lightweight local machine-learning safety scorer
- An AI-agent response layer powered by LangChain + Groq when configured
- ESP8266 device registration, heartbeat, and alert scaffolding

## Frontend Routes

- `GET /health`
- `GET /v1/users/{user_id}`
- `PUT /v1/users/{user_id}`
- `GET /v1/contacts/{user_id}`
- `POST /v1/map/bootstrap-nearby`
- `GET /v1/map/{user_id}/locations`
- `POST /v1/map/location`
- `GET /v1/chat/{user_id}`
- `POST /v1/chat`
- `POST /v1/ml/safety-score`
- `POST /v1/ai/respond`
- `POST /v1/agent/respond`
- `POST /v1/device/alert`
- `POST /v1/device/esp8266/register`
- `POST /v1/device/esp8266/heartbeat`
- `GET /v1/device/esp8266/status/{user_id}`

## Environment

Copy `.env.example` to `.env` if you want to override local settings:

- `NEON_DATABASE_URL` controls the Neon/Postgres connection string
- `DATABASE_URL` works as a fallback if you prefer that variable name
- `SAFEHER_CONTACT_DISTANCE_M` controls how far the seeded Father, Mother, and Police contacts are placed from the current map center
- `GROQ_API_KEY` enables the LangChain/Groq-backed assistant replies
- `SAFEHER_GROQ_MODEL` selects the model used for replies, defaulting to `llama-3.3-70b-versatile`
- `SAFEHER_GROQ_TEMPERATURE` controls how creative replies are, defaulting to `0.2`
- `APP_NAME` is a friendly label for the backend

The emergency contacts are placed about 500m from the user's current map location when the frontend calls `POST /v1/map/bootstrap-nearby`.

## AI Usage

The safest way to use AI in the project is to let the backend call Groq through LangChain and keep the frontend on the same API routes it already uses.

Example request for the generic assistant endpoint:

```bash
curl -X POST http://localhost:8000/v1/ai/respond \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "demo-user",
    "mode": "chat",
    "message": "I feel unsafe walking home",
    "context": {
      "location_present": true
    }
  }'
```

Use `mode: "map"` for route and ETA guidance, `mode: "contacts"` for emergency contact help, and `mode: "device"` for ESP8266 or alert flows.
