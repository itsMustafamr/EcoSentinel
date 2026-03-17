"""
EcoSentinel — Main Application
Gradio UI + FastAPI REST endpoints (consumed by the OpenClaw skill).
"""

import json
import os
import sys
import threading

import gradio as gr
import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Local modules
sys.path.insert(0, os.path.dirname(__file__))
from data_engine import EcoSentinelEngine
from llm_client  import ask_nemotron, format_risk_context, format_city_context

# ── Startup ─────────────────────────────────────────────────────────────────

print("[EcoSentinel] Initialising data engine...")
engine = EcoSentinelEngine()
CITIES = ["All"] + engine.get_cities()

# ── FastAPI (REST API for OpenClaw skill) ────────────────────────────────────

api = FastAPI(title="EcoSentinel API")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    context: str = ""


@api.get("/api/top-risk")
def api_top_risk(city: str = None, n: int = 5):
    df = engine.top_risk(city=city, n=n)
    context = format_risk_context(df)
    return {"context": context, "data": df.fillna("").to_dict("records")}


@api.get("/api/city-summary")
def api_city_summary():
    df = engine.city_summary()
    context = format_city_context(df)
    return {"context": context, "data": df.fillna("").to_dict("records")}


@api.post("/api/ask")
def api_ask(req: AskRequest):
    answer = ask_nemotron(req.question, req.context)
    return {"answer": answer}


@api.get("/api/health")
def api_health():
    return {"status": "ok", "businesses": len(engine.risk_table)}


def _run_api():
    uvicorn.run(api, host="0.0.0.0", port=7861, log_level="warning")

threading.Thread(target=_run_api, daemon=True).start()
print("[EcoSentinel] REST API running on :7861")

# ── Helper: build Folium map ─────────────────────────────────────────────────

def build_map(city_filter: str = "All") -> str:
    city_df = engine.city_summary()

    if city_filter and city_filter != "All":
        city_df = city_df[city_df["CITY"].str.upper() == city_filter.upper()]
        center_lat = city_df["lat"].mean() if not city_df.empty else 37.35
        center_lon = city_df["lon"].mean() if not city_df.empty else -121.95
        zoom = 13
    else:
        center_lat, center_lon, zoom = 37.35, -121.95, 10

    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom,
                   tiles="CartoDB positron")

    for _, row in city_df.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["lon"]):
            continue
        risk = float(row["avg_risk"])
        # Colour: red = high risk (low score), green = safe
        colour = (
            "#d62728" if risk < -0.15 else
            "#ff7f0e" if risk < 0.0  else
            "#2ca02c"
        )
        radius = max(6, min(22, int(row["business_count"] / 5)))
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=radius,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.7,
            popup=folium.Popup(
                f"<b>{row['CITY']}</b><br>"
                f"Businesses: {int(row['business_count'])}<br>"
                f"Avg Score: {row['avg_score']:.1f}<br>"
                f"Total Critical Violations: {int(row['total_critical'])}<br>"
                f"Risk Score: {risk:.3f}",
                max_width=220,
            ),
            tooltip=row["CITY"],
        ).add_to(m)

    # Legend
    legend_html = """
    <div style='position:fixed;bottom:30px;left:30px;z-index:9999;
                background:white;padding:10px;border-radius:8px;
                border:1px solid #ccc;font-size:12px;'>
        <b>EcoSentinel Risk</b><br>
        <span style='color:#d62728'>&#9679;</span> High Risk (score &lt; -0.15)<br>
        <span style='color:#ff7f0e'>&#9679;</span> Moderate Risk<br>
        <span style='color:#2ca02c'>&#9679;</span> Lower Risk
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m._repr_html_()

# ── Gradio UI ────────────────────────────────────────────────────────────────

def update_map(city):
    return build_map(city)

def get_risk_table(city, n):
    df = engine.top_risk(city=city, n=int(n)).fillna("")
    df = df.rename(columns={
        "name": "Business",
        "CITY": "City",
        "avg_score": "Avg Score",
        "total_critical": "Critical Violations",
        "fail_rate": "Fail Rate",
        "inspection_count": "Inspections",
        "last_inspection": "Last Inspection",
        "risk_score": "Risk Score",
    })
    return df

def get_violations_chart():
    df = engine.violation_type_summary()
    fig = px.bar(
        df.head(15),
        x="count",
        y="DESCRIPTION",
        orientation="h",
        color="critical_pct",
        color_continuous_scale=["green", "orange", "red"],
        labels={"count": "Occurrences", "DESCRIPTION": "", "critical_pct": "Critical %"},
        title="Top 15 Most Common Violations",
        height=500,
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, margin=dict(l=10))
    return fig

def get_city_chart():
    df = engine.city_summary()
    fig = px.scatter(
        df,
        x="avg_score",
        y="total_critical",
        size="business_count",
        color="avg_risk",
        color_continuous_scale=["red", "orange", "green"],
        hover_name="CITY",
        labels={
            "avg_score": "Average Inspection Score",
            "total_critical": "Total Critical Violations",
            "avg_risk": "Risk Score",
            "business_count": "# Businesses",
        },
        title="City Risk Profile: Score vs Critical Violations",
        height=460,
    )
    return fig

def respond(message, history):
    """Gradio ChatInterface handler."""
    # Detect city in message
    cities = engine.get_cities()
    mentioned_city = next(
        (c for c in cities if c.upper() in message.upper()), None
    )
    df = engine.top_risk(city=mentioned_city, n=5)
    context = format_risk_context(df)

    # Add city summary if asking about comparison/overview
    if any(kw in message.lower() for kw in
           ["compare", "city", "cities", "worst", "best", "overview", "summary"]):
        city_df = engine.city_summary()
        context += "\n\nCity-level summary:\n" + format_city_context(city_df)

    return ask_nemotron(message, context, history)


# ── Gradio layout ─────────────────────────────────────────────────────────────

css = """
.header { text-align: center; padding: 10px 0 0 0; }
.header h1 { font-size: 2em; margin-bottom: 0; }
.header p  { color: #666; margin-top: 4px; }
"""

with gr.Blocks(title="EcoSentinel", theme=gr.themes.Soft(), css=css) as demo:

    gr.HTML("""
    <div class='header'>
      <h1>🌿 EcoSentinel</h1>
      <p>AI-Powered Environmental Health Intelligence &nbsp;·&nbsp; Santa Clara County
         &nbsp;·&nbsp; Powered by NVIDIA DGX Spark (GB10) &nbsp;·&nbsp; RAPIDS &nbsp;·&nbsp; Nemotron &nbsp;·&nbsp; OpenClaw</p>
    </div>
    """)

    with gr.Tab("🗺️ Risk Map"):
        city_map_dd = gr.Dropdown(
            choices=CITIES, value="All", label="Filter by City", scale=1
        )
        map_html = gr.HTML(value=build_map())
        city_map_dd.change(update_map, city_map_dd, map_html)

    with gr.Tab("📊 Risk Dashboard"):
        with gr.Row():
            city_dd  = gr.Dropdown(choices=CITIES, value="All", label="City")
            n_slider = gr.Slider(5, 50, value=15, step=5, label="Show top N")
        risk_tbl = gr.Dataframe(
            value=get_risk_table("All", 15),
            interactive=False,
        )
        city_dd.change(get_risk_table,  [city_dd, n_slider], risk_tbl)
        n_slider.change(get_risk_table, [city_dd, n_slider], risk_tbl)

    with gr.Tab("📈 Violation Insights"):
        with gr.Row():
            viol_chart = gr.Plot(value=get_violations_chart())
            city_chart = gr.Plot(value=get_city_chart())

    with gr.Tab("🤖 AI Assistant (Nemotron)"):
        gr.Markdown(
            "Ask anything about food safety in Santa Clara County. "
            "Powered by **Nemotron** running on the DGX Spark.\n\n"
            "**Try:** *Which city has the worst critical violation rate?* · "
            "*Show me the highest-risk restaurants in San Jose* · "
            "*What are the most dangerous violation types?*"
        )
        gr.ChatInterface(fn=respond)

    with gr.Tab("🔗 OpenClaw Integration"):
        gr.Markdown("""
## Live via OpenClaw

EcoSentinel is connected to **OpenClaw** — query it from any messaging platform:

| Platform | Status |
|----------|--------|
| WhatsApp / Telegram / Signal | via OpenClaw skill |
| Slack / Discord / Teams | via OpenClaw skill |
| Web Chat | this dashboard |

### Example queries (send from your phone):
- *"Which restaurants in Milpitas have the most critical violations?"*
- *"Is Santa Clara safer than Sunnyvale for food safety?"*
- *"Show me restaurants in San Jose that failed inspections"*
- *"What are the most dangerous violation types in the county?"*

### OpenClaw REST API (used by the skill):
```
GET  http://localhost:7861/api/top-risk?city=<city>&n=<n>
GET  http://localhost:7861/api/city-summary
POST http://localhost:7861/api/ask  { "question": "...", "context": "..." }
GET  http://localhost:7861/api/health
```

OpenClaw Gateway: `ws://localhost:18789`
        """)

# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[EcoSentinel] Launching Gradio on :7860")
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
