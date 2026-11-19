
import asyncio
import hmac
import logging
import time

from dataclasses import dataclass
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os

import torch

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

Token = os.environ.get("Token", "")


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = Path("/app/model")
MODEL_NAME = "floxoris/harmony-v1.2"

TOXIC_THRESHOLD = 0.65

MAX_TEXT_LENGTH = 1000
MAX_TOKENS = 128

MAX_QUEUE_SIZE = 10
MAX_WAIT_SECONDS = 180

# ==========================================
# LOGGING
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("harmony")


# ==========================================
# PYTORCH SETTINGS
# ==========================================

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


# ==========================================
# GLOBAL STATE
# ==========================================

tokenizer = None
model = None

request_queue = None
inference_executor = None
worker_task = None

ready = False
accepting_requests = False

processed_requests = 0
failed_requests = 0

STOP_WORKER = object()


# ==========================================
# JOB STRUCTURE
# ==========================================

@dataclass
class Job:
    text: str
    future: asyncio.Future
    created_at: float


# ==========================================
# MODEL LOADING
# ==========================================

def load_model():

    global tokenizer, model

    logger.info("Loading model: %s", MODEL_NAME)

    if not MODEL_PATH.exists():
        raise RuntimeError("Model directory not found")

    if not (MODEL_PATH / "config.json").exists():
        raise RuntimeError("config.json not found")

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True
    )

    model.eval()

    if model.config.num_labels != 2:
        raise RuntimeError(
            f"Expected 2 classes, got {model.config.num_labels}"
        )

    logger.info("Model loaded successfully")

    logger.info(
        "Parameters: %s",
        sum(p.numel() for p in model.parameters())
    )


# ==========================================
# CLASSIFICATION
# ==========================================

def classify_text(text: str):

    started = time.perf_counter()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        padding=False
    )

    with torch.inference_mode():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    safe_score = float(probabilities[0].item())
    toxic_score = float(probabilities[1].item())

    label = (
        "toxic"
        if toxic_score >= TOXIC_THRESHOLD
        else "safe"
    )

    latency = (
        time.perf_counter() - started
    ) * 1000

    return {
        "model": MODEL_NAME,
        "label": label,
        "class": 1 if label == "toxic" else 0,
        "safe_score": round(safe_score, 4),
        "toxic_score": round(toxic_score, 4),
        "threshold": TOXIC_THRESHOLD,
        "latency_ms": round(latency, 2)
    }


# ==========================================
# SINGLE WORKER
# ==========================================

async def queue_worker():

    global processed_requests
    global failed_requests

    loop = asyncio.get_running_loop()

    logger.info("Queue worker started")

    while True:

        job = await request_queue.get()

        try:

            if job is STOP_WORKER:
                return

            # Клиент мог отменить ожидание
            if job.future.cancelled():
                continue

            logger.info(
                "Processing request | queue=%s",
                request_queue.qsize()
            )

            result = await loop.run_in_executor(
                inference_executor,
                classify_text,
                job.text
            )

            processed_requests += 1

            if not job.future.done():
                job.future.set_result(result)

            logger.info(
                "Completed | inference=%.2f ms",
                result["latency_ms"]
            )

        except Exception as error:

            failed_requests += 1

            logger.exception(
                "Inference error: %s",
                error
            )

            if not job.future.done():
                job.future.set_exception(
                    RuntimeError("Model inference failed")
                )

        finally:
            if job is not STOP_WORKER:
                request_queue.task_done()


# ==========================================
# APPLICATION LIFESPAN
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global request_queue
    global inference_executor
    global worker_task
    global ready
    global accepting_requests
    global model
    global tokenizer

    if not isinstance(Token, str) or len(Token) < 32:
        raise RuntimeError("API token must contain at least 32 characters")

    logger.info("Starting Harmony API")

    # Load model once
    load_model()

    # Create limited queue
    request_queue = asyncio.Queue(
        maxsize=MAX_QUEUE_SIZE
    )

    # Only one inference thread
    inference_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="harmony"
    )

    # Start one worker
    worker_task = asyncio.create_task(
        queue_worker()
    )

    ready = True
    accepting_requests = True

    logger.info("Harmony API READY")

    try:
        yield

    finally:

        logger.info("Graceful shutdown started")

        ready = False
        accepting_requests = False

        # Reject queued jobs during shutdown
        while not request_queue.empty():

            try:
                job = request_queue.get_nowait()

                if job is not STOP_WORKER:
                    if not job.future.done():
                        job.future.set_exception(
                            RuntimeError("Server shutting down")
                        )

                request_queue.task_done()

            except asyncio.QueueEmpty:
                break

        # Stop worker after current inference
        await request_queue.put(STOP_WORKER)

        await worker_task

        # Close executor
        inference_executor.shutdown(
            wait=True,
            cancel_futures=True
        )

        model = None
        tokenizer = None

        logger.info("Harmony API stopped")


# ==========================================
# FASTAPI
# ==========================================

app = FastAPI(
    title="Harmony Moderation API",
    version="3.0.0",
    lifespan=lifespan
)


# ==========================================
# INPUT VALIDATION
# ==========================================

class CheckRequest(BaseModel):

    text: str = Field(
        min_length=1,
        max_length=MAX_TEXT_LENGTH
    )


# ==========================================
# ROOT
# ==========================================

@app.get("/")
async def root():

    return {
        "service": "Harmony Moderation API",
        "model": MODEL_NAME,
        "status": "online" if ready else "starting"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
async def health():

    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "online" if ready else "starting",
            "model_loaded": model is not None,
            "queue_size": (
                request_queue.qsize()
                if request_queue is not None
                else 0
            ),
            "queue_limit": MAX_QUEUE_SIZE,
            "processed": processed_requests,
            "failed": failed_requests
        }
    )


# ==========================================
# CLASSIFICATION API
# ==========================================

@app.post("/check")
async def check_text(
    data: CheckRequest,
    x_api_token: str = Header(default="")
):

    # Authorization
    if not hmac.compare_digest(x_api_token, Token):

        raise HTTPException(
            status_code=401,
            detail="Invalid API token"
        )

    # Server readiness
    if not ready or not accepting_requests:

        raise HTTPException(
            status_code=503,
            detail="Server unavailable"
        )

    # Empty text
    if not data.text.strip():

        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty"
        )

    loop = asyncio.get_running_loop()

    future = loop.create_future()

    job = Job(
        text=data.text,
        future=future,
        created_at=time.monotonic()
    )

    # Add to FIFO queue
    try:

        request_queue.put_nowait(job)

    except asyncio.QueueFull:

        raise HTTPException(
            status_code=429,
            detail="Queue full. Retry later.",
            headers={"Retry-After": "3"}
        )

    # Wait for classification
    try:

        result = await asyncio.wait_for(
            asyncio.shield(future),
            timeout=MAX_WAIT_SECONDS
        )

        return result

    except asyncio.TimeoutError:

        future.cancel()

        raise HTTPException(
            status_code=504,
            detail="Processing timeout"
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Classification failed"
        )
