# 🌿 EcoSentinel

**AI-powered environmental health intelligence for Santa Clara County — running 100% locally on NVIDIA DGX Spark GB10.**

> Ask your Telegram bot *"Which restaurants in San Jose are highest risk?"* and get a live, Nemotron-powered answer in seconds. No cloud. No external APIs. Everything runs on the DGX Spark.

Built for the **NVIDIA Hack for Impact** hackathon — Environmental Impact track + OpenClaw bounty.

---

## Demo

Send a message to **@EcoSentinel_dgx_bot** on Telegram:

```
Which restaurants in San Jose are highest risk?
Show me the worst restaurants in Sunnyvale
What are the most common violations in Santa Clara County?
Compare food safety across all cities
```

---

## What It Does

EcoSentinel ingests Santa Clara County's public food inspection dataset (8,500+ businesses, 183,000+ violations), scores every restaurant by health risk using GPU-accelerated analytics, and makes that intelligence accessible via a Telegram bot — powered entirely by **Nemotron-Nano-3-30B** running on the DGX Spark.

- Real-time food safety risk ranking for 8,579 businesses across 12 cities
- Natural language queries via Telegram (`@EcoSentinel_dgx_bot`)
- Interactive web dashboard with risk maps, charts, and AI assistant
- 100% local inference — no OpenAI, no Anthropic, no cloud APIs

---

## Architecture

```
Telegram User
      │
      ▼
OpenClaw Gateway (port 18789)
      │
      └─ Embedded Agent → Anthropic Proxy (port 8090)
                                │
                                ├─ Food query? → REST API → inject live context
                                └─ Nemotron via llama-server /completion
                                                        │
                                                        ▼
                                            Nemotron-Nano-3-30B-A3B
                                            (GGUF Q4_K_M, 99 GPU layers)
                                            NVIDIA GB10 Blackwell · sm_121

Data Pipeline:
SCC Open Data (CSV) → RAPIDS cuDF / pandas → Risk Scoring → FastAPI REST (port 7861)
                                                           → Gradio UI  (port 7860)
```

---

## Tech Stack

| Component | Technology | Port |
|-----------|-----------|------|
| LLM Inference | llama-server + Nemotron-Nano-3-30B (Q4_K_M GGUF) | 8080 |
| Telegram Bot | OpenClaw + `openclaw-ecosentinel` plugin | 18789 |
| Anthropic Proxy | FastAPI — routes OpenClaw agent → Nemotron with live data context | 8090 |
| Data Engine | RAPIDS cuDF / pandas fallback | — |
| REST API | FastAPI | 7861 |
| Web Dashboard | Gradio | 7860 |
| Data Visualization | Folium (maps) + Plotly (charts) | — |
| GPU Container | RAPIDS Notebooks Docker | 8888 |

---

## Dataset

- **Source:** Santa Clara County Department of Environmental Health (public open data)
- **Businesses:** 8,579 | **Inspections:** 83,774 | **Violations:** 183,851
- **Cities:** San Jose, Santa Clara, Sunnyvale, Milpitas, Campbell, Cupertino, Mountain View, Los Altos, Los Gatos, Saratoga, Morgan Hill, Gilroy

### Risk Score Formula

```
risk_score = (avg_score/100 × 0.4) - (critical_density/10 × 0.4) - (fail_rate × 0.2)
```

---

## Project Structure

```
EcoSentinel/
├── app/
│   ├── data_engine.py       # RAPIDS/pandas GPU risk scoring engine
│   ├── llm_client.py        # llama-server /completion client (ChatML)
│   ├── main.py              # Gradio UI + FastAPI REST API
│   └── anthropic_proxy.py   # Anthropic API → Nemotron proxy (with live data injection)
├── openclaw_skill/
│   ├── ecosentinel.js       # OpenClaw plugin (Telegram message handler)
│   ├── openclaw.plugin.json # Plugin manifest
│   └── package.json
├── .openclaw/
│   └── openclaw.json.example # OpenClaw config template
├── docker-compose.yml       # RAPIDS container
├── startup.sh               # Full stack startup script
└── README.md
```

---

## Setup

### Prerequisites

- NVIDIA DGX Spark GB10 (ARM64, sm_121, CUDA 13.0)
- llama-server built from llama.cpp
- Nemotron-Nano-3-30B-A3B Q4_K_M GGUF model
- Docker, OpenClaw, Node.js 22, Python 3.11

### Quick Start

```bash
git clone https://github.com/itsMustafamr/EcoSentinel.git
cd EcoSentinel
python3 -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn requests gradio folium plotly pandas

# Configure OpenClaw (add your Telegram bot token)
cp .openclaw/openclaw.json.example ~/.openclaw/openclaw.json

# Place SCC CSV data in ~/data/ (businesses.csv, inspections.csv, violations.csv)

chmod +x startup.sh && ./startup.sh
```

---

## API

```bash
# Health check
curl http://localhost:7861/api/health

# Top risk businesses
curl "http://localhost:7861/api/top-risk?city=San+Jose&n=5"

# Ask Nemotron
curl -X POST http://localhost:7861/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which restaurants should I avoid in Sunnyvale?"}'
```

---

## Why DGX Spark GB10

| Challenge | How GB10 Solved It |
|-----------|-------------------|
| 30B model + full dataset simultaneously | 128GB unified memory — no swapping |
| Fast inference | Blackwell GPU (sm_121), 99 layers offloaded |
| Data privacy | 100% local — health data never leaves the machine |
| Concurrent requests | 2 parallel KV cache slots, 20K tokens each |

---

## Hackathon

**NVIDIA Hack for Impact** · Environmental Impact track · OpenClaw bounty  
**Team:** Mohammed Musthafa Rafi · Built in ~3 hours on DGX Spark GB10

---

## License

MIT
