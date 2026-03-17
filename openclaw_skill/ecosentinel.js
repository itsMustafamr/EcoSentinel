/**
 * EcoSentinel — OpenClaw Plugin (CommonJS)
 *
 * Intercepts ALL Telegram messages and routes them through Nemotron
 * running locally on the DGX Spark GB10. No external API needed.
 *
 * - Food safety queries → Nemotron + live SCC inspection data context
 * - All other queries   → Nemotron directly (general assistant mode)
 */

"use strict";

const ECOSENTINEL_API = process.env.ECOSENTINEL_API || "http://localhost:7861/api";
const LLAMA_SERVER    = process.env.LLAMA_SERVER    || "http://localhost:8080";

const KNOWN_CITIES = [
  "San Jose", "Santa Clara", "Sunnyvale", "Milpitas", "Campbell",
  "Cupertino", "Mountain View", "Los Altos", "Los Gatos", "Saratoga",
  "Morgan Hill", "Gilroy",
];

const FOOD_TRIGGERS = [
  "food safety", "food inspection", "restaurant", "health inspection",
  "violation", "ecosentinel", "health score", "food risk",
  "sanitation", "critical violation", "which restaurant",
];

function extractCity(text) {
  for (const city of KNOWN_CITIES) {
    if (text.toLowerCase().includes(city.toLowerCase())) return city;
  }
  return null;
}

function isFoodQuery(text) {
  const lower = text.toLowerCase();
  return FOOD_TRIGGERS.some(t => lower.includes(t));
}

// ── Ask Nemotron directly via llama-server /completion endpoint ───────────

async function askNemotron(question, context) {
  const system =
    "You are EcoSentinel, an AI assistant running on an NVIDIA DGX Spark GB10 supercomputer " +
    "in Santa Clara County, California. You help with environmental health, food safety, and " +
    "general questions. Be concise and helpful. When answering food safety questions, cite " +
    "specific business names and scores from the data context if provided.";

  const userContent = context
    ? `Data context:\n${context}\n\nQuestion: ${question}`
    : question;

  const prompt =
    `<|im_start|>system\n${system}<|im_end|>\n` +
    `<|im_start|>user\n${userContent}<|im_end|>\n` +
    `<|im_start|>assistant\n`;

  const res = await fetch(`${LLAMA_SERVER}/completion`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      n_predict: 512,
      temperature: 0.1,
      top_p: 0.9,
      stop: ["<|im_end|>", "<|im_start|>"],
    }),
  });

  if (!res.ok) throw new Error(`llama-server HTTP ${res.status}`);
  const data = await res.json();
  let content = (data.content || "").trim();
  // Strip thinking tags if present
  if (content.includes("<think>") && content.includes("</think>")) {
    content = content.split("</think>").pop().trim();
  }
  return content;
}

// ── Main message handler ──────────────────────────────────────────────────

async function handleMessage(message, actions) {
  const text = (message.text || message.content || "").trim();
  if (!text) return;

  let context = "";

  try {
    if (isFoodQuery(text)) {
      // Inject live food inspection data from the RAPIDS engine
      const city = extractCity(text);
      const url = city
        ? `${ECOSENTINEL_API}/top-risk?city=${encodeURIComponent(city)}&n=5`
        : `${ECOSENTINEL_API}/top-risk?n=5`;

      const dataRes = await fetch(url);
      if (dataRes.ok) context = (await dataRes.json()).context || "";

      if (/compar|overview|worst|best|all cities/i.test(text)) {
        const cityRes = await fetch(`${ECOSENTINEL_API}/city-summary`);
        if (cityRes.ok) context += "\n\nCity summary:\n" + ((await cityRes.json()).context || "");
      }
    }

    const answer = await askNemotron(text, context);
    await actions.reply(
      `🌿 *EcoSentinel* — powered by Nemotron on NVIDIA DGX Spark\n\n${answer}`
    );
  } catch (err) {
    await actions.reply(
      `⚠️ Nemotron unreachable. Check llama-server at ${LLAMA_SERVER}/health`
    );
  }
}

// ── OpenClaw plugin registration ──────────────────────────────────────────

function activate(ctx) {
  const { hooks, logger } = ctx || {};
  if (logger) logger.info("[EcoSentinel] Plugin activated — all messages → Nemotron on DGX Spark");

  if (hooks && hooks.onMessage) {
    // Handle every incoming message — no Claude agent needed
    hooks.onMessage(function (msg, actions) {
      return handleMessage(msg, actions);
    });
  }
}

function register(ctx) { activate(ctx); }

module.exports = { register, activate, handleMessage };
