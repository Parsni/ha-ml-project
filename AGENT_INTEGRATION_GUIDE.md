# Home Assistant ML Model Integration Guide

This guide explains how to integrate the Home Assistant Next-Action Predictor model as a tool-calling agent or as a direct Home Assistant automation service.

## Recommended pattern

Use the model as a small local HTTP API service (for example, FastAPI) that accepts structured features and returns a prediction JSON. Home Assistant or a tool-calling agent can call the API when needed.

This is the best option because it keeps the model reusable, testable, and observable while keeping camera data local.

### Best practice
- Prefer sending detections/features instead of raw camera frames.
- Keep ML inference local to the Home Assistant LAN if possible.
- Define a clear prediction schema and confidence threshold.
- Use the model as a tool first; use direct execution only when you want model-driven automation.

## Architecture

Camera (Frigate / OpenCV / YOLO) -> Vision pipeline -> Feature extractor -> ML prediction service -> Action decision -> Home Assistant automation/service call

## Example request payload

```json
{
  "timestamp": "2026-10-16T12:00:00Z",
  "camera_id": "front_door",
  "detections": [
    {"label": "person", "bbox": [100, 50, 60, 140], "confidence": 0.92}
  ],
  "ha_states": {
    "light.front_porch": "off",
    "sensor.front_door_motion": false
  },
  "recent_actions": ["notify_homeowner"],
  "sensor_features": {
    "motion": true
  }
}
```

## Example response payload

```json
{
  "predicted_next_action": "turn_on_front_porch_light",
  "action_params": {"entity_id": "light.front_porch"},
  "confidence": 0.87,
  "explainability": {
    "feature_importance": {
      "motion": 0.5,
      "person_presence": 0.3
    }
  },
  "latency_ms": 45
}
```

## Why use it as a tool instead of embedding it directly in the agent?

### Tool pattern (recommended)
Pros:
- Clear separation of concerns
- Easier to test and monitor
- Reusable by Home Assistant, LLM agents, and other services
- Simpler security model

Cons:
- Adds network/API overhead
- Requires a small API wrapper

### Agent pattern (model directly executes actions)
Pros:
- Simpler end-to-end flow
- Less glue code in Home Assistant

Cons:
- More security complexity
- Harder to reason about and debug
- More coupling between the model and Home Assistant state

## Recommended setup

Use the model as a local service behind a small API, and keep the actual decision logic in Home Assistant or a coordinator agent.

That means:
- Model predicts next action
- Agent or Home Assistant decides whether to execute it
- Confidence threshold decides whether action is safe to run

## Minimal FastAPI example

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn
import requests
from typing import Any, Dict, List, Optional

class Detection(BaseModel):
    label: str
    bbox: List[int]
    confidence: float

class PredictRequest(BaseModel):
    timestamp: str
    camera_id: str
    detections: List[Detection]
    ha_states: Dict[str, Any] = {}
    recent_actions: List[str] = []
    sensor_features: Dict[str, Any] = {}

class PredictResponse(BaseModel):
    predicted_next_action: str
    action_params: Dict[str, Any]
    confidence: float
    explainability: Optional[Dict[str, Any]] = None
    latency_ms: Optional[int] = None

app = FastAPI(title="HA Next-Action Predictor")
MODEL_PATH = "model.joblib"
model = joblib.load(MODEL_PATH)


def featurize(req: PredictRequest):
    has_person = any(d.label == "person" for d in req.detections)
    motion = bool(req.sensor_features.get("motion", False))
    return [[int(has_person), int(motion)]]


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    X = featurize(payload)
    try:
        probs = model.predict_proba(X)[0]
        pred_idx = probs.argmax()
        pred_label = model.classes_[pred_idx]
        confidence = float(probs[pred_idx])
    except Exception:
        pred_label = str(model.predict(X)[0])
        confidence = 1.0

    action_map = {
        "turn_on_front_porch_light": {"entity_id": "light.front_porch"},
    }

    return PredictResponse(
        predicted_next_action=pred_label,
        action_params=action_map.get(pred_label, {}),
        confidence=confidence,
        explainability={"note": "implement SHAP or feature_importance here"},
        latency_ms=0,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## Home Assistant integration examples

### Option 1: HA automation calls the model (recommended)

Use a REST command, then let Home Assistant decide what to do.

```yaml
rest_command:
  ml_predict:
    url: "http://ml-host:8000/predict"
    method: POST
    content_type: "application/json"
    payload: >
      {
        "timestamp": "{{ now().isoformat() }}",
        "camera_id": "front_door",
        "detections": [{"label":"person","bbox":[100,50,60,140],"confidence":0.92}],
        "ha_states": {{ states() | tojson }},
        "sensor_features": {"motion": true}
      }
```

Then trigger an automation based on motion, call the model, and act on the returned output.

### Option 2: Model calls Home Assistant automatically

Use a second endpoint that calls a Home Assistant service after prediction.

```python
HA_API = "http://homeassistant.local:8123/api"
HA_TOKEN = "LONG_LIVED_ACCESS_TOKEN"

@app.post("/predict_and_act", response_model=PredictResponse)
def predict_and_act(payload: PredictRequest):
    resp = predict(payload)
    if resp.confidence > 0.8:
        headers = {
            "Authorization": f"Bearer {HA_TOKEN}",
            "Content-Type": "application/json",
        }
        url = f"{HA_API}/services/light/turn_on"
        body = {"entity_id": resp.action_params.get("entity_id")}
        r = requests.post(url, json=body, headers=headers, timeout=3)
        if r.status_code >= 400:
            raise HTTPException(status_code=500, detail="HA service call failed")
    return resp
```

This approach is useful if you want the model to act directly, but it is less conservative than letting HA decide.

## What the model should output

A good output schema is:

```json
{
  "predicted_next_action": "turn_on_front_porch_light",
  "action_params": {"entity_id": "light.front_porch"},
  "confidence": 0.87,
  "explainability": {"feature_importance": {"motion": 0.5}},
  "latency_ms": 45
}
```

## Security and privacy guidance

- Keep raw camera frames local.
- Send only object detections, features, or embeddings.
- Authenticate your API endpoint with a secret/API key.
- Use TLS if the service is exposed beyond the local network.
- Keep HA tokens in environment variables or secret storage.
- Add rate limiting and request logging.

## Performance guidance

- For latency-critical automations, use compact feature vectors instead of raw image data.
- Run the vision pipeline separately (Frigate/OpenCV/YOLO) and pass only detections to the model.
- If needed, optimize inference with ONNX or compiled inference libraries.
- Add health checks and metrics for model latency and prediction confidence.

## When to use a custom component

Use a custom Home Assistant component only if you need:
- very tight state integration,
- continual stateful prediction sensors,
- lower-latency local execution,
- no external service dependency.

Otherwise, prefer a local model service + automation integration because it is simpler and safer.

## Bottom line

The correct pattern is:

- Use the model as a local inference tool
- Feed it structured features not raw images
- Let Home Assistant or the agent decide whether to execute the action
- Use confidence thresholds and action safety checks

This gives you a practical, real-world Home Assistant agent setup that can work with vision inputs while still staying safe and reproducible.

## Suggested next step

If you want, the next step is to implement a real, runnable version of this architecture in the repo:
1. Add a FastAPI inference service
2. Add a Dockerfile
3. Add a Home Assistant automation example
4. Add a feature extraction script that turns camera detections into model inputs
5. Add a simple test payload and prediction example

This is the most useful practical next step for a Home Assistant ML agent integration.


