
import json
import re

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma4:e4b"  # swap for gemma4:e2b (lighter) or gemma4:12b/26b/31b (stronger)

app = FastAPI(title="Study Buddy API")

# Allow the local frontend (opened as a file or served on any port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def call_gemma(messages: list[dict], images: list[str] | None = None) -> str:
    """Send a chat request to the local Gemma 4 model via Ollama and return the text reply."""
    if images:
        # Ollama expects base64 image data attached to the last user message.
        messages[-1]["images"] = images

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "messages": messages, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Can't reach Ollama at localhost:11434. Is ollama serve running?",
        )
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Gemma 4 took too long to respond.")

    data = resp.json()
    return data["message"]["content"]


def extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from a model response."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise HTTPException(status_code=502, detail="Model did not return valid JSON.")
    return json.loads(match.group(0))


# ---------- /api/explain ----------

class ExplainRequest(BaseModel):
    question: str = ""
    image_base64: str | None = None  # raw base64, no data: prefix


@app.post("/api/explain")
def explain(req: ExplainRequest):
    if not req.question and not req.image_base64:
        raise HTTPException(status_code=400, detail="Provide a question or an image.")

    system_prompt = (
        "You are a patient, encouraging tutor for school-age students. "
        "When given a problem (as text and/or a photo), do NOT just give the final answer. "
        "Walk through the reasoning step by step, in plain language, then state the final "
        "answer clearly at the end labeled 'Answer:'. Keep steps short and numbered."
    )
    user_text = req.question or "Here is a photo of a problem. Please explain how to solve it."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
    ]
    images = [req.image_base64] if req.image_base64 else None
    explanation = call_gemma(messages, images)
    return {"explanation": explanation}
