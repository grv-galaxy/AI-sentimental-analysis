---
title: SentimentAI
emoji: 🎭
colorFrom: purple
colorTo: blue
sdk: docker
app_file: app.py
pinned: false
license: apache-2.0
short_description: "3-class sentiment analysis API — airzipm"
---

# 🎭 SentimentAI — RoBERTa Sentiment Analysis API

A production-grade, high-concurrency sentiment analysis API built on
[airzipm/sentiment-analysis-roberta](https://huggingface.co/airzipm/sentiment-analysis-roberta).

Classifies text into **Positive**, **Neutral**, or **Negative** with confidence scores.

---

## 🚀 Quick Start

### Live Demo
Open the Space URL in your browser — the frontend (`index.html`) loads automatically.

### Base URL
```
https://airzipm-sentimentai.hf.space
```

---

## 📡 API Endpoints

| Method | Path | Rate Limit | Description |
|--------|------|-----------|-------------|
| `GET` | `/` | — | Serve frontend `index.html` |
| `GET` | `/health` | — | Model readiness & server stats |
| `GET` | `/ping` | — | Lightweight liveness check |
| `POST` | `/analyze` | 30/min per IP | Single text sentiment analysis |
| `POST` | `/batch` | 10/min per IP | Batch analysis (up to 10 texts) |
| `GET` | `/docs` | — | Interactive Swagger UI |
| `GET` | `/redoc` | — | ReDoc API documentation |

---

## 🔍 Endpoint Reference

### `GET /ping`
Ultra-lightweight liveness probe. No inference. Always < 5ms.
Frontend pings this every 30 seconds to prevent the Space from sleeping.

**Response:**
```json
{
  "pong": true,
  "model_ready": true,
  "t": 1717500000.123
}
```

---

### `GET /health`
Model readiness check with server statistics.
Always returns HTTP 200 — read the `status` field to check readiness.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu",
  "model_name": "airzipm/sentiment-analysis-roberta",
  "uptime_s": 342.7,
  "requests_served": 1284,
  "version": "1.0.0"
}
```

| `status` value | Meaning |
|---------------|---------|
| `"ok"` | Model loaded, ready to serve |
| `"loading"` | Model is still initializing (cold start) |
| `"error"` | Model failed to load (check logs) |

---

### `POST /analyze`
Analyze sentiment of a single text. HuggingFace Inference API compatible.

**Rate limit:** 30 requests per IP per minute

**Request body:**
```json
{
  "inputs": "This movie was absolutely amazing!"
}
```

**Successful response (200):**
```json
{
  "label": "Positive",
  "score": 0.973241,
  "all_scores": [
    {"label": "Positive", "score": 0.973241},
    {"label": "Neutral",  "score": 0.021034},
    {"label": "Negative", "score": 0.005725}
  ],
  "response_ms": 87
}
```

**Error responses:**

| Status | When | Body |
|--------|------|------|
| 400 | Text empty or > 2000 chars | `{"detail": "validation error..."}` |
| 429 | Rate limit hit | `{"detail": "Rate limit exceeded"}` |
| 503 | Model still loading | `{"detail": "Model is still loading..."}` |
| 500 | Unexpected error | `{"detail": "Inference failed..."}` |

---

### `POST /batch`
Analyze up to 10 texts in a single efficient request.
All texts are processed in one RoBERTa forward pass (padded batch).

**Rate limit:** 10 requests per IP per minute (batch is more expensive)

**Request body:**
```json
{
  "inputs": [
    "I loved this product!",
    "The service was average.",
    "Absolutely terrible experience."
  ]
}
```

**Successful response (200):**
```json
{
  "results": [
    {
      "label": "Positive",
      "score": 0.971203,
      "all_scores": [
        {"label": "Positive", "score": 0.971203},
        {"label": "Neutral",  "score": 0.022411},
        {"label": "Negative", "score": 0.006386}
      ],
      "response_ms": 134
    },
    {
      "label": "Neutral",
      "score": 0.683441,
      "all_scores": [
        {"label": "Neutral",  "score": 0.683441},
        {"label": "Positive", "score": 0.201233},
        {"label": "Negative", "score": 0.115326}
      ],
      "response_ms": 134
    },
    {
      "label": "Negative",
      "score": 0.941872,
      "all_scores": [
        {"label": "Negative", "score": 0.941872},
        {"label": "Neutral",  "score": 0.042311},
        {"label": "Positive", "score": 0.015817}
      ],
      "response_ms": 134
    }
  ],
  "batch_size": 3,
  "total_ms": 148
}
```

---

## 🖥️ Code Examples

### JavaScript (Fetch API)

```javascript
const API = "https://airzipm-sentimentai.hf.space";

// ── Single analysis ──────────────────────────────────────────────────────────
async function analyze(text) {
  const response = await fetch(`${API}/analyze`, {
    method : "POST",
    headers: { "Content-Type": "application/json" },
    body   : JSON.stringify({ inputs: text }),
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.detail || "Request failed");
  }

  return response.json();
  // → { label: "Positive", score: 0.973, all_scores: [...], response_ms: 87 }
}

// ── Batch analysis ───────────────────────────────────────────────────────────
async function analyzeBatch(texts) {
  const response = await fetch(`${API}/batch`, {
    method : "POST",
    headers: { "Content-Type": "application/json" },
    body   : JSON.stringify({ inputs: texts }),
  });
  return response.json();
  // → { results: [...], batch_size: 3, total_ms: 148 }
}

// ── Keep-alive ping (prevents HF Space from sleeping) ─────────────────────
setInterval(async () => {
  const { model_ready } = await fetch(`${API}/ping`).then(r => r.json());
  console.log("Model ready:", model_ready);
}, 30_000);
```

### Python (httpx)

```python
import httpx

API = "https://airzipm-sentimentai.hf.space"

# Single text
response = httpx.post(f"{API}/analyze", json={"inputs": "This is great!"})
result = response.json()
print(result["label"], result["score"])  # → Positive 0.973

# Batch
batch_resp = httpx.post(f"{API}/batch", json={
    "inputs": ["Loved it!", "Meh.", "Terrible."]
})
for r in batch_resp.json()["results"]:
    print(r["label"], r["score"])
```

### cURL

```bash
# Single analysis
curl -X POST "https://airzipm-sentimentai.hf.space/analyze" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "This product completely exceeded my expectations!"}'

# Batch analysis
curl -X POST "https://airzipm-sentimentai.hf.space/batch" \
  -H "Content-Type: application/json" \
  -d '{"inputs": ["Amazing!", "It was okay.", "Terrible experience."]}'

# Health check
curl "https://airzipm-sentimentai.hf.space/health"

# Ping
curl "https://airzipm-sentimentai.hf.space/ping"
```

---

## ⚡ Architecture & Concurrency

### How it handles multiple simultaneous users

```
User 1  → acquires semaphore slot 1 → running inference (~100ms)
User 2  → acquires semaphore slot 2 → running inference
User 3  → acquires semaphore slot 3 → running inference
User 4  → acquires semaphore slot 4 → running inference
User 5  → WAITS in asyncio queue  (non-blocking — event loop stays free)
User 6  → WAITS in asyncio queue
...
User 20 → WAITS in asyncio queue

When User 1 finishes:
  → releases slot
  → User 5 immediately acquires it and begins inference
```

**No requests are dropped or rejected.** The asyncio event loop stays free to
accept new connections and serve `/ping` responses while inferences run.

### Component Breakdown

| Component | Role |
|-----------|------|
| **FastAPI** | Async request routing, Pydantic validation |
| **Gunicorn + 4 UvicornWorkers** | Multi-process concurrency (~100+ connections) |
| **asyncio.Semaphore(4)** | Caps simultaneous inferences to prevent OOM |
| **loop.run_in_executor** | Runs CPU-bound inference off the event loop |
| **slowapi** | Per-IP rate limiting (in-memory, no Redis) |
| **RoBERTa (loaded once)** | Weights in RAM at startup — never reloaded per request |

### Response Headers

Every response includes:

| Header | Value | Use |
|--------|-------|-----|
| `X-Response-Time` | `87ms` | Frontend displays this |
| `X-Model-Ready` | `true` / `false` | Frontend status bar |
| `Access-Control-Allow-Origin` | `*` | CORS — any frontend domain |

---

## 🔧 Local Development

```bash
# Clone and install
git clone https://huggingface.co/spaces/airzipm/sentimentai
cd sentimentai
pip install -r requirements.txt

# Run locally (single worker, port 7860)
python app.py

# Or with gunicorn (production-like, 4 workers)
gunicorn app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:7860 \
  --timeout 120

# Visit: http://localhost:7860
# Swagger: http://localhost:7860/docs
```

---

## ⚠️ Cold Start Note

This Space runs on the **free tier**, which means:
- The Space **sleeps** after ~5 minutes of inactivity
- The first request after sleeping triggers a **cold start** (20–50 seconds)
- The model is downloaded (~500MB) and loaded into RAM on first start
- The frontend handles this gracefully with a "warming up" status bar

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE)

Model: [airzipm/sentiment-analysis-roberta](https://huggingface.co/airzipm/sentiment-analysis-roberta)

---

<!--
OPTION A — Gradio SDK wrapper (simpler, no Dockerfile needed)
If you prefer not to use Docker, you can wrap FastAPI with Gradio:

README frontmatter:
  sdk: gradio
  sdk_version: "4.36.1"

Add to requirements.txt:
  gradio==4.36.1

Add to bottom of app.py (before uvicorn.run):
  import gradio as gr
  gr_app = gr.Blocks(title="SentimentAI")
  with gr_app:
      gr.HTML("""
        <h2>🎭 SentimentAI API</h2>
        <p>REST API is running. Visit <a href="/docs">/docs</a> for Swagger UI
        or use the <a href="/">frontend</a>.</p>
      """)
  app = gr.mount_gradio_app(app, gr_app, path="/ui")

Tradeoff: Option A is simpler but has Gradio overhead.
Option B (Dockerfile, used above) is cleaner and production-grade.
-->
