"""
SentimentAI — Production FastAPI inference server
Model: airzipm/sentiment-analysis-roberta
Author: airzipm | huggingface.co/airzipm

Run locally:  python app.py
Run via gunicorn (prod):
  gunicorn app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:7860 --timeout 120 --keepalive 5
"""

# ── SECTION A — IMPORTS ──────────────────────────────────────────────────────

import os
import time
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

# ── SECTION B — LOGGING SETUP ────────────────────────────────────────────────

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sentiment_api")

# ── SECTION C — CONFIG & GLOBAL STATE ────────────────────────────────────────

MODEL_NAME   = "airzipm/sentiment-analysis-roberta"
MAX_LENGTH   = 128       # must match training tokenizer config
MAX_TEXT_LEN = 2000      # hard cap on input characters
MAX_BATCH    = 10        # max texts per /batch request
MAX_WORKERS  = 4         # semaphore: max simultaneous inferences
DEVICE_NAME  = "cuda" if torch.cuda.is_available() else "cpu"

# TODO: If the model is ever made private, set HF_TOKEN here:
# HF_TOKEN = os.getenv("HF_TOKEN", None)
# Pass token=HF_TOKEN to from_pretrained() calls below.

# Label mapping — must match the model's config.json id2label
ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Module-level state — populated during lifespan startup
tokenizer           : Optional[AutoTokenizer]                          = None
model               : Optional[AutoModelForSequenceClassification]     = None
device              : Optional[torch.device]                           = None
inference_semaphore : Optional[asyncio.Semaphore]                      = None
startup_time        : Optional[float]                                  = None
request_count       : int                                              = 0
model_loaded        : bool                                             = False

# ── SECTION D — MODEL LOADING (lifespan) ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.
    Loads model once at startup and frees memory at shutdown.
    Using lifespan instead of deprecated @app.on_event decorators.
    """
    # ── STARTUP ──────────────────────────────────────────────────────────────
    global tokenizer, model, device, inference_semaphore
    global startup_time, model_loaded

    startup_time = time.time()
    logger.info("=" * 60)
    logger.info("Starting SentimentAI API server...")
    logger.info(f"Device: {DEVICE_NAME}")
    logger.info(f"Model:  {MODEL_NAME}")
    logger.info("=" * 60)

    t0 = time.time()
    try:
        device = torch.device(DEVICE_NAME)

        # Load tokenizer (fast Rust-based tokenizer for speed)
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            use_fast = True,    # Rust tokenizer — ~10× faster than Python
        )

        # Load model weights
        logger.info("Loading model weights...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            # FP16 on GPU halves memory & doubles speed; FP32 on CPU (no FP16 benefit there)
            torch_dtype    = torch.float16 if DEVICE_NAME == "cuda" else torch.float32,
            low_cpu_mem_usage = True,   # stream weights to CPU in chunks (lower peak RAM)
        )
        model = model.to(device)
        model.eval()    # CRITICAL: disables dropout layers for deterministic inference

        # Warmup: run one dummy inference to JIT-compile any CUDA kernels.
        # Without this, the first real request takes 2–5× longer (cold CUDA path).
        logger.info("Running warmup inference...")
        dummy_inputs = tokenizer(
            "warmup",
            return_tensors = "pt",
            max_length     = 16,
            truncation     = True,
            padding        = True,
        )
        dummy_inputs = {k: v.to(device) for k, v in dummy_inputs.items()}
        with torch.no_grad():
            _ = model(**dummy_inputs)

        # asyncio.Semaphore MUST be created inside an async context (here, inside lifespan)
        # so it's bound to the correct running event loop.
        inference_semaphore = asyncio.Semaphore(MAX_WORKERS)

        load_ms = int((time.time() - t0) * 1000)
        model_loaded = True
        logger.info(f"✅ Model ready in {load_ms}ms")

    except Exception as exc:
        # Don't crash the server — let /health report the failure state
        logger.error(f"FATAL: Model loading failed: {exc}", exc_info=True)
        model_loaded = False

    yield   # ← server is live and handling requests here

    # ── SHUTDOWN ─────────────────────────────────────────────────────────────
    logger.info("Shutting down — releasing model memory...")
    if model is not None:
        del model
        if DEVICE_NAME == "cuda":
            torch.cuda.empty_cache()    # return VRAM to OS
    logger.info("Shutdown complete.")

# ── SECTION E — APP INITIALIZATION ───────────────────────────────────────────

# slowapi rate limiter — uses client IP as the key, in-memory store (no Redis needed)
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title       = "SentimentAI API",
    description = (
        "3-class sentiment analysis powered by fine-tuned RoBERTa. "
        "Returns Positive / Neutral / Negative with confidence scores. "
        "By airzipm | huggingface.co/airzipm"
    ),
    version     = "1.0.0",
    docs_url    = "/docs",      # Swagger UI
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)

# Attach rate limiter + its 429 error handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS: allow all origins so the index.html can call from any domain
# (GitHub Pages, HF Spaces, localhost — all work without proxy)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],          # any origin
    allow_credentials = False,          # must be False when allow_origins=["*"]
    allow_methods     = ["GET", "POST", "OPTIONS"],
    allow_headers     = ["*"],
    max_age           = 3600,           # browsers cache preflight for 1 hour
)

# ── SECTION F — PYDANTIC SCHEMAS ────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    inputs: str = Field(
        ...,
        min_length  = 1,
        max_length  = MAX_TEXT_LEN,
        description = "Text to analyze (1–2000 characters)",
        examples    = ["This movie was absolutely amazing!"],
    )

    @field_validator("inputs")
    @classmethod
    def strip_and_check(cls, v: str) -> str:
        """Strip whitespace and reject blank strings."""
        v = v.strip()
        if len(v) == 0:
            raise ValueError("Text cannot be empty or whitespace only")
        return v


class SentimentScore(BaseModel):
    label : str
    score : float


class AnalyzeResponse(BaseModel):
    """
    HuggingFace Inference API-compatible response format.
    all_scores matches the HF pipeline output list format.
    """
    label       : str                   # "Positive" | "Neutral" | "Negative"
    score       : float                 # confidence of top label (0–1)
    all_scores  : list[SentimentScore]  # all 3 classes sorted by score desc
    response_ms : int                   # server-side inference time in ms


class BatchRequest(BaseModel):
    inputs: list[str] = Field(
        ...,
        min_length  = 1,
        max_length  = MAX_BATCH,
        description = f"List of texts to analyze (1–{MAX_BATCH} items)",
    )

    @field_validator("inputs")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        """Strip each text and validate length."""
        cleaned = []
        for i, text in enumerate(texts):
            t = text.strip()
            if len(t) == 0:
                raise ValueError(f"Item {i} is empty after stripping whitespace")
            if len(t) > MAX_TEXT_LEN:
                raise ValueError(f"Item {i} exceeds {MAX_TEXT_LEN} character limit")
            cleaned.append(t)
        return cleaned


class BatchResponse(BaseModel):
    results    : list[AnalyzeResponse]
    batch_size : int
    total_ms   : int


class HealthResponse(BaseModel):
    status          : str    # "ok" | "loading" | "error"
    model_loaded    : bool
    device          : str
    model_name      : str
    uptime_s        : float
    requests_served : int
    version         : str


# ── SECTION G — INFERENCE ENGINE ────────────────────────────────────────────

def _run_inference_sync(texts: list[str]) -> list[dict]:
    """
    Synchronous inference function — runs in a thread pool executor
    (via asyncio.loop.run_in_executor) so it doesn't block the event loop.

    Accepts a list of texts (batch), tokenizes them together with padding,
    runs one forward pass through RoBERTa, applies softmax, and returns
    per-text predictions sorted by confidence descending.

    Returns:
        List of dicts: [{"label": str, "score": float, "all_scores": [...]}]
    """
    global model, tokenizer, device

    # Tokenize all texts as a batch.
    # padding=True pads shorter sequences to the longest in this batch
    # (more efficient than padding everything to MAX_LENGTH).
    inputs = tokenizer(
        texts,
        return_tensors = "pt",
        max_length     = MAX_LENGTH,
        truncation     = True,          # silently truncate texts > MAX_LENGTH tokens
        padding        = True,          # pad to longest in batch
    )
    # Move input tensors to the target device (CPU or GPU)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():   # no gradient tracking — saves memory & time
        if DEVICE_NAME == "cuda":
            # Automatic mixed precision on GPU: FP16 math, FP32 accumulation
            # Roughly 2× throughput on modern NVIDIA GPUs
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(**inputs).logits
        else:
            logits = model(**inputs).logits     # shape: [batch_size, num_labels]

    # Softmax converts raw logits → probabilities that sum to 1.0
    probs = torch.softmax(logits, dim=-1).cpu().numpy()   # shape: [batch_size, 3]

    results = []
    for prob_row in probs:
        # Build list of {label, score} for each class
        all_scores = [
            {"label": ID2LABEL[i], "score": round(float(prob_row[i]), 6)}
            for i in range(len(prob_row))
        ]
        # Sort descending by score — HF Inference API convention
        all_scores.sort(key=lambda x: x["score"], reverse=True)
        results.append({
            "label"      : all_scores[0]["label"],  # top predicted class
            "score"      : all_scores[0]["score"],  # its confidence
            "all_scores" : all_scores,
        })

    return results


async def run_inference(texts: list[str]) -> tuple[list[dict], int]:
    """
    Async wrapper around _run_inference_sync.

    Concurrency model (what happens when 20 users call simultaneously):

        User 1  → acquires semaphore slot 1 → running inference
        User 2  → acquires semaphore slot 2 → running inference
        User 3  → acquires semaphore slot 3 → running inference
        User 4  → acquires semaphore slot 4 → running inference
        User 5  → WAITS in asyncio queue (non-blocking — event loop is free)
        User 6  → WAITS in asyncio queue
        ...
        User 20 → WAITS in asyncio queue

        When User 1 finishes (~100ms on CPU):
            → releases slot 1
            → User 5 immediately acquires it
            → begins inference

    No request is dropped. No request blocks the server.
    The event loop stays free to accept new connections and serve /ping
    while existing requests wait. Estimated CPU throughput: ~40 req/sec.

    Args:
        texts: list of pre-validated, stripped strings

    Returns:
        (results, elapsed_ms) where results is a list of prediction dicts
    """
    global inference_semaphore, request_count

    if not model_loaded:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "Model is still loading. Please retry in a few seconds.",
        )

    t0 = time.perf_counter()

    # Semaphore ensures at most MAX_WORKERS concurrent inferences.
    # Excess requests yield control back to the event loop (non-blocking await).
    async with inference_semaphore:
        loop    = asyncio.get_event_loop()
        # run_in_executor moves the CPU-bound sync function to a thread pool,
        # freeing the event loop to handle other requests during inference.
        results = await loop.run_in_executor(None, _run_inference_sync, texts)

    elapsed_ms    = int((time.perf_counter() - t0) * 1000)
    request_count += len(texts)     # approximate; fine for stats (not atomically safe)
    return results, elapsed_ms


# ── SECTION H — MIDDLEWARE ───────────────────────────────────────────────────

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """
    Adds X-Response-Time and X-Model-Ready headers to every response.
    Frontend reads X-Response-Time to display accurate response times.
    """
    t0       = time.perf_counter()
    response = await call_next(request)
    ms       = int((time.perf_counter() - t0) * 1000)
    response.headers["X-Response-Time"] = f"{ms}ms"
    response.headers["X-Model-Ready"]   = str(model_loaded).lower()
    return response


# ── SECTION H1 — ROOT: SERVE INDEX.HTML ─────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_index():
    """
    Serve the frontend index.html if present in the working directory.
    HF Spaces copies all repo files to /app — so index.html lands here.
    Falls back to a JSON API info card if no index.html exists.
    """
    for candidate in ["/app/index.html", "index.html", "./index.html"]:
        if os.path.exists(candidate):
            return FileResponse(candidate, media_type="text/html")

    # Fallback: API info as JSON (useful during development)
    return JSONResponse({
        "name"       : "SentimentAI API",
        "version"    : "1.0.0",
        "model"      : MODEL_NAME,
        "author"     : "airzipm",
        "endpoints"  : {
            "POST /analyze" : "Single text sentiment analysis",
            "POST /batch"   : "Batch analysis (up to 10 texts)",
            "GET  /health"  : "Model readiness & stats",
            "GET  /ping"    : "Lightweight liveness check",
            "GET  /docs"    : "Interactive Swagger UI",
        },
    })


# ── SECTION H2 — HEALTH CHECK ───────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns model readiness and server stats.

    Always returns HTTP 200 (never 503) — the frontend reads the
    'status' field to determine if the model is ready.
    Returning 503 here would confuse browser error-handling logic.
    """
    return HealthResponse(
        status          = "ok" if model_loaded else "loading",
        model_loaded    = model_loaded,
        device          = DEVICE_NAME,
        model_name      = MODEL_NAME,
        uptime_s        = round(time.time() - startup_time, 1) if startup_time else 0.0,
        requests_served = request_count,
        version         = "1.0.0",
    )


# ── SECTION H3 — SINGLE TEXT ANALYSIS ───────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("30/minute")     # 30 requests per IP per minute
async def analyze(request: Request, body: AnalyzeRequest):
    """
    Analyze the sentiment of a single text.

    Request body (HuggingFace Inference API compatible):
        {"inputs": "your text here"}

    Response:
        {
            "label"      : "Positive",
            "score"      : 0.973241,
            "all_scores" : [
                {"label": "Positive", "score": 0.973241},
                {"label": "Neutral",  "score": 0.021034},
                {"label": "Negative", "score": 0.005725}
            ],
            "response_ms": 87
        }

    Errors:
        400 — empty or too-long text (caught by Pydantic)
        429 — rate limit exceeded (30/min per IP)
        503 — model not yet loaded
        500 — unexpected inference error
    """
    t_start = time.perf_counter()
    text    = body.inputs
    ip      = get_remote_address(request)

    logger.info(f"POST /analyze | len={len(text)} | ip={ip}")

    try:
        results, infer_ms = await run_inference([text])
        r        = results[0]
        total_ms = int((time.perf_counter() - t_start) * 1000)

        logger.info(
            f"  → {r['label']} ({r['score']:.3f}) | "
            f"infer={infer_ms}ms total={total_ms}ms"
        )

        return AnalyzeResponse(
            label       = r["label"],
            score       = r["score"],
            all_scores  = [SentimentScore(**s) for s in r["all_scores"]],
            response_ms = infer_ms,
        )

    except HTTPException:
        raise   # re-raise 503/429/400 as-is
    except Exception as exc:
        logger.error(f"  ✗ Inference error: {exc}", exc_info=True)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = "Inference failed. Please try again.",
        )


# ── SECTION H4 — BATCH ANALYSIS ─────────────────────────────────────────────

@app.post("/batch", response_model=BatchResponse)
@limiter.limit("10/minute")     # stricter — batch is ~N× more expensive
async def analyze_batch(request: Request, body: BatchRequest):
    """
    Analyze sentiment of up to 10 texts in a single request.

    All texts are tokenized together and run through RoBERTa in one
    forward pass (via padding), making this much more efficient than
    10 separate /analyze calls.

    Request body:
        {"inputs": ["text 1", "text 2", "text 3"]}

    Response:
        {
            "results"    : [{...}, {...}, {...}],
            "batch_size" : 3,
            "total_ms"   : 145
        }

    Errors:
        400 — batch > 10, or any text is empty/too long
        429 — rate limit exceeded (10/min per IP)
        503 — model not yet loaded
        500 — inference error
    """
    texts    = body.inputs
    n        = len(texts)
    t_start  = time.perf_counter()
    ip       = get_remote_address(request)

    logger.info(f"POST /batch | n={n} | ip={ip}")

    try:
        results, infer_ms = await run_inference(texts)
        total_ms = int((time.perf_counter() - t_start) * 1000)

        logger.info(f"  → batch n={n} | infer={infer_ms}ms | total={total_ms}ms")

        return BatchResponse(
            results    = [
                AnalyzeResponse(
                    label       = r["label"],
                    score       = r["score"],
                    all_scores  = [SentimentScore(**s) for s in r["all_scores"]],
                    response_ms = infer_ms,
                )
                for r in results
            ],
            batch_size = n,
            total_ms   = total_ms,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"  ✗ Batch error: {exc}", exc_info=True)
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = "Batch inference failed. Please try again.",
        )


# ── SECTION H5 — PING (lightweight keep-alive) ───────────────────────────────

@app.get("/ping")
async def ping():
    """
    Ultra-lightweight liveness endpoint. Never runs inference.
    Frontend calls this every 30 seconds to prevent the HF Space
    from going to sleep due to inactivity (free tier sleeps after ~5min).

    No rate limit applied — this must always respond instantly.
    Expected response time: < 5ms.
    """
    return {
        "pong"        : True,
        "model_ready" : model_loaded,
        "t"           : time.time(),
    }


# ── SECTION H6 — GLOBAL EXCEPTION HANDLER ───────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all for any unhandled exception that bubbles up.
    Always returns structured JSON — never an HTML error page.
    Logs full traceback for debugging.
    """
    logger.error(
        f"Unhandled error | {request.method} {request.url.path} | {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code = 500,
        content     = {
            "error"  : "Internal server error",
            "detail" : "An unexpected error occurred. Please try again.",
            "path"   : str(request.url.path),
        },
    )


# ── SECTION I — ENTRY POINT ──────────────────────────────────────────────────

if __name__ == "__main__":
    # Local development: single worker, reload on file changes
    # Production (HF Space): use gunicorn command above
    uvicorn.run(
        "app:app",
        host      = "0.0.0.0",
        port      = 7860,           # HF Spaces always uses 7860
        workers   = 1,              # 1 worker when running directly
        log_level = "info",
        access_log= True,
        reload    = False,          # set True for local dev hot-reload
    )
