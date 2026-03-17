"""
Anthropic API Proxy → Nemotron (llama-server)

Accepts requests in Anthropic Messages API format from OpenClaw,
translates them to llama-server /completion calls, and returns
proper Anthropic-format responses.

OpenClaw believes it's talking to Claude.
It's actually talking to Nemotron-30B on the DGX Spark GB10.

For food-safety queries, live SCC inspection data is injected
as context before the Nemotron call — so answers are grounded
in real data, not generic knowledge.
"""

import json
import re
import time
import uuid
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

LLAMA_SERVER    = "http://localhost:8080"
ECOSENTINEL_API = "http://localhost:7861/api"
TIMEOUT         = 120

app = FastAPI(title="Anthropic→Nemotron Proxy")

# ── Food query detection ──────────────────────────────────────────────────

FOOD_TRIGGERS = [
    "food safety", "food inspection", "restaurant", "health inspection",
    "violation", "ecosentinel", "health score", "food risk",
    "sanitation", "critical violation", "which restaurant", "worst restaurant",
    "best restaurant", "avoid", "highest risk", "safest",
]

KNOWN_CITIES = [
    "San Jose", "Santa Clara", "Sunnyvale", "Milpitas", "Campbell",
    "Cupertino", "Mountain View", "Los Altos", "Los Gatos", "Saratoga",
    "Morgan Hill", "Gilroy",
]

def is_food_query(text: str) -> bool:
    lower = text.lower()
    return any(t in lower for t in FOOD_TRIGGERS)

def extract_city(text: str):
    for city in KNOWN_CITIES:
        if city.lower() in text.lower():
            return city
    return None

def fetch_food_context(text: str) -> str:
    """Fetch live SCC inspection context for food queries."""
    try:
        city = extract_city(text)
        url = (
            f"{ECOSENTINEL_API}/top-risk?city={requests.utils.quote(city)}&n=5"
            if city else
            f"{ECOSENTINEL_API}/top-risk?n=5"
        )
        r = requests.get(url, timeout=10)
        ctx = r.json().get("context", "") if r.ok else ""

        if re.search(r'compar|overview|worst|best|all cities', text, re.I):
            r2 = requests.get(f"{ECOSENTINEL_API}/city-summary", timeout=10)
            if r2.ok:
                ctx += "\n\nCity summary:\n" + (r2.json().get("context", ""))
        return ctx
    except Exception:
        return ""

# ── Prompt building ───────────────────────────────────────────────────────

ECOSENTINEL_SYSTEM = (
    "You are EcoSentinel, an AI assistant running on an NVIDIA DGX Spark GB10 supercomputer "
    "in Santa Clara County, California. You help with environmental health, food safety, and "
    "general questions. Be concise and helpful. When answering food safety questions, cite "
    "specific business names and scores from the data context if provided. "
    "Always answer directly from the data provided — do not say you cannot access live data."
)

def build_chatml_prompt(messages: list, system: str = "", data_context: str = "") -> str:
    # Use EcoSentinel system prompt, appending any data context
    effective_system = ECOSENTINEL_SYSTEM
    if data_context:
        effective_system += f"\n\nLive inspection data:\n{data_context}"

    prompt = f"<|im_start|>system\n{effective_system}<|im_end|>\n"

    for msg in messages:
        role = msg.get("role", "user")
        chatml_role = "assistant" if role == "assistant" else "user"
        content = msg.get("content", "")
        if isinstance(content, list):
            text_parts = [
                c.get("text", "") for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            content = "\n".join(text_parts)
        if content:
            prompt += f"<|im_start|>{chatml_role}\n{content}<|im_end|>\n"

    prompt += "<|im_start|>assistant\n"
    return prompt


def call_nemotron(prompt: str, max_tokens: int = 1024) -> str:
    resp = requests.post(
        f"{LLAMA_SERVER}/completion",
        json={
            "prompt": prompt,
            "n_predict": min(max_tokens, 1024),
            "temperature": 0.1,
            "top_p": 0.9,
            "stop": ["<|im_end|>", "<|im_start|>"],
        },
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    content = resp.json().get("content", "").strip()
    if "<think>" in content and "</think>" in content:
        content = content.split("</think>")[-1].strip()
    return content


# ── API endpoints ─────────────────────────────────────────────────────────

@app.post("/v1/messages")
async def messages(request: Request):
    """Anthropic Messages API endpoint."""
    body = await request.json()

    messages_list = body.get("messages", [])
    max_tokens    = body.get("max_tokens", 1024)
    stream        = body.get("stream", False)
    model         = body.get("model", "claude-opus-4-6")

    # Extract the latest user message text for food-query detection
    last_user_text = ""
    for msg in reversed(messages_list):
        if msg.get("role") == "user":
            c = msg.get("content", "")
            if isinstance(c, list):
                c = " ".join(x.get("text","") for x in c if isinstance(x,dict) and x.get("type")=="text")
            last_user_text = c
            break

    # Inject food context if this is a food-safety query
    data_context = ""
    if is_food_query(last_user_text):
        data_context = fetch_food_context(last_user_text)

    prompt = build_chatml_prompt(messages_list, data_context=data_context)

    try:
        content = call_nemotron(prompt, max_tokens)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    if stream:
        def generate():
            yield "event: message_start\ndata: " + json.dumps({
                "type": "message_start",
                "message": {"id": msg_id, "type": "message", "role": "assistant",
                            "content": [], "model": model,
                            "stop_reason": None, "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0}}
            }) + "\n\n"
            yield "event: content_block_start\ndata: " + json.dumps({
                "type": "content_block_start", "index": 0,
                "content_block": {"type": "text", "text": ""}
            }) + "\n\n"
            chunk_size = 20
            for i in range(0, len(content), chunk_size):
                yield "event: content_block_delta\ndata: " + json.dumps({
                    "type": "content_block_delta", "index": 0,
                    "delta": {"type": "text_delta", "text": content[i:i+chunk_size]}
                }) + "\n\n"
            yield "event: content_block_stop\ndata: " + json.dumps({
                "type": "content_block_stop", "index": 0
            }) + "\n\n"
            yield "event: message_delta\ndata: " + json.dumps({
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": len(content.split())}
            }) + "\n\n"
            yield "event: message_stop\ndata: " + json.dumps({
                "type": "message_stop"
            }) + "\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")

    return JSONResponse({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content}],
        "model": model,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": len(prompt.split()),
            "output_tokens": len(content.split()),
        },
    })


@app.get("/v1/models")
async def list_models():
    return JSONResponse({
        "data": [
            {"id": "claude-opus-4-6",   "object": "model"},
            {"id": "claude-sonnet-4-6", "object": "model"},
            {"id": "nemotron-30b",       "object": "model"},
        ]
    })


@app.get("/health")
async def health():
    return {"status": "ok", "backend": "nemotron-30b@dgx-spark"}


if __name__ == "__main__":
    print("[Proxy] Anthropic → Nemotron proxy on :8090")
    print("[Proxy] Food queries → live SCC context + Nemotron")
    print("[Proxy] All inference runs on Nemotron-30B @ DGX Spark GB10")
    uvicorn.run(app, host="0.0.0.0", port=8090, log_level="info")
