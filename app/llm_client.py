"""
EcoSentinel LLM Client
Uses llama-server's /completion endpoint (ChatML format) with the
Nemotron-Nano-3-30B GGUF model running on the DGX Spark GB10.
"""

import requests
import pandas as pd

# llama-server running on port 8080 with Nemotron GGUF
LLAMA_SERVER_URL = "http://localhost:8080/completion"
TIMEOUT = 90

SYSTEM_PROMPT = (
    "You are EcoSentinel, an environmental health AI assistant for Santa Clara County, California. "
    "You help health inspectors, city planners, and the public understand food safety risks from real inspection data. "
    "You have data on 8,500+ food businesses, 83,000+ inspections, and 183,000+ violations. "
    "Be concise. Cite business names and scores when relevant. "
    "Risk score: lower = more dangerous. Critical violations = immediate health hazard."
)


def _build_chatml_prompt(system: str, user: str, history: list = None) -> str:
    """Build a ChatML-format prompt string for llama-server /completion endpoint."""
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n"
    if history:
        for user_msg, assistant_msg in history[-2:]:
            prompt += f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{assistant_msg}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    return prompt


def ask_nemotron(question: str, data_context: str = "", history: list = None) -> str:
    """
    Query Nemotron with optional data context and conversation history.
    Returns the assistant's response as a string.
    """
    if data_context:
        user_content = f"Data context:\n{data_context}\n\nQuestion: {question}"
    else:
        user_content = question

    prompt = _build_chatml_prompt(SYSTEM_PROMPT, user_content, history)

    try:
        resp = requests.post(
            LLAMA_SERVER_URL,
            json={
                "prompt": prompt,
                "n_predict": 512,
                "temperature": 0.1,
                "top_p": 0.9,
                "stop": ["<|im_end|>", "<|im_start|>"],
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        content = resp.json().get("content", "").strip()
        # Strip residual thinking tags that Nemotron may emit
        if "<think>" in content and "</think>" in content:
            content = content.split("</think>")[-1].strip()
        return content
    except requests.exceptions.ConnectionError:
        return (
            "[EcoSentinel] Nemotron (llama-server) is not reachable at localhost:8080. "
            "Restart: cd /home/nvidia/llama.cpp && "
            "LD_LIBRARY_PATH=build/lib ./build/bin/llama-server "
            "--model /home/nvidia/models/gguf/ggml-org--Nemotron-Nano-3-30B-A3B-GGUF/Nemotron-Nano-3-30B-A3B-Q4_K_M.gguf "
            "--port 8080 --n-gpu-layers 99"
        )
    except Exception as e:
        return f"[EcoSentinel] LLM error: {e}"


def format_risk_context(df: pd.DataFrame, max_rows: int = 8) -> str:
    """Convert a risk dataframe into a compact text block for the LLM prompt."""
    if df is None or df.empty:
        return "No data available."

    lines = []
    for _, row in df.head(max_rows).iterrows():
        last = row.get("last_inspection", "")
        if pd.notna(last) and last != "":
            try:
                last = pd.Timestamp(last).strftime("%Y-%m-%d")
            except Exception:
                last = str(last)
        lines.append(
            f"- {row.get('name','?')} ({row.get('CITY','?')}): "
            f"avg_score={row.get('avg_score', 0):.1f}, "
            f"critical_violations={int(row.get('total_critical', 0))}, "
            f"fail_rate={row.get('fail_rate', 0):.0%}, "
            f"inspections={int(row.get('inspection_count', 0))}, "
            f"last_inspected={last}"
        )
    return "\n".join(lines)


def format_city_context(df: pd.DataFrame) -> str:
    """Format city-level summary for LLM context."""
    if df is None or df.empty:
        return "No city data available."
    lines = []
    for _, row in df.iterrows():
        lines.append(
            f"- {row['CITY']}: avg_score={row['avg_score']:.1f}, "
            f"total_critical={int(row['total_critical'])}, "
            f"businesses={int(row['business_count'])}, "
            f"risk_score={row['avg_risk']:.3f}"
        )
    return "\n".join(lines)
