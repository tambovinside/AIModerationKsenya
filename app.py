import asyncio
import hmac
import json
import logging
import os
import time
import urllib.error
import urllib.request

from dataclasses import dataclass
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# CONFIGURATION
# ============================================================

API_TOKEN = os.environ.get("Token", "").strip()

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/model"))
MODEL_NAME = "Aveniats/Tocxic-moderation"
MODEL_VERSION = "RuToxic multi-label"

# Thresholds are configurable in Render Environment.
# Profanity/obscenity is deliberately NOT included in toxic class.
INSULT_THRESHOLD = float(os.environ.get("INSULT_THRESHOLD", "0.70"))
THREAT_THRESHOLD = float(os.environ.get("THREAT_THRESHOLD", "0.65"))
DANGEROUS_THRESHOLD = float(os.environ.get("DANGEROUS_THRESHOLD", "0.75"))

MAX_TEXT_LENGTH = 1000
MAX_TOKENS = 128
MAX_QUEUE_SIZE = 10
MAX_WAIT_SECONDS = 180
DOWNLOAD_TIMEOUT_SECONDS = 90
DOWNLOAD_RETRIES = 3

# Files are downloaded from this repository on first startup.
# model.safetensors was listed twice in the supplied URLs; it is downloaded once.
MODEL_FILES = {
    "config.json": "https://huggingface.co/Aveniats/Tocxic-moderation/resolve/main/config.json?download=true",
    "db.py": "https://huggingface.co/Aveniats/Tocxic-moderation/resolve/main/db.py?download=true",
    "model.safetensors": "https://huggingface.co/Aveniats/Tocxic-moderation/resolve/main/model.safetensors?download=true",
    "special_tokens_map.json": "https://huggingface.co/Aveniats/Tocxic-moderation/resolve/main/special_tokens_map.json?download=true",
    "tokenizer.json": "https://huggingface.co/Aveniats/Tocxic-moderation/resolve/main/tokenizer.json?download=true",
    "tokenizer_config.json": "https://huggingface.co/Aveniats/Tocxic-moderation/resolve/main/tokenizer_config.json?download=true",
    "vocab.txt": "https://huggingface.co/Aveniats/Tocxic-moderation/resolve/main/vocab.txt?download=true",
}

EXPECTED_LABELS = {
    0: "non-toxic",
    1: "insult",
    2: "obscenity",
    3: "threat",
    4: "dangerous",
}

# ============================================================
# LOGGING / PYTORCH
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("rutoxic")

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

# ============================================================
# GLOBAL STATE
# ============================================================

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


@dataclass
class Job:
    text: str
    future: asyncio.Future
    created_at: float


# ============================================================
# MODEL DOWNLOAD
# ============================================================

def download_file(filename: str, url: str) -> None:
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    target = MODEL_PATH / filename

    # Reuse a complete non-empty file across restarts on persistent disks.
    if target.is_file() and target.stat().st_size > 0:
        logger.info("Model file already exists: %s (%s bytes)", filename, target.stat().st_size)
        return

    temporary = target.with_suffix(target.suffix + ".part")
    headers = {
        "User-Agent": "RuToxic-Moderation-Server/1.0",
        "Accept": "*/*",
    }

    last_error = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            logger.info("Downloading %s (attempt %s/%s)", filename, attempt, DOWNLOAD_RETRIES)
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise RuntimeError(f"HTTP {status} while downloading {filename}")

                with open(temporary, "wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)

            if not temporary.exists() or temporary.stat().st_size == 0:
                raise RuntimeError(f"Downloaded file is empty: {filename}")

            temporary.replace(target)
            logger.info("Downloaded %s (%s bytes)", filename, target.stat().st_size)
            return

        except Exception as error:
            last_error = error
            logger.warning("Download failed for %s: %s", filename, error)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(min(attempt * 2, 5))

    raise RuntimeError(f"Could not download {filename}: {last_error}")


def ensure_model_files() -> None:
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    for filename, url in MODEL_FILES.items():
        download_file(filename, url)

    required = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ]
    missing = [
        name for name in required
        if not (MODEL_PATH / name).is_file() or (MODEL_PATH / name).stat().st_size == 0
    ]
    if missing:
        raise RuntimeError("Required model files are missing: " + ", ".join(missing))

    # Check that this repository actually contains the expected five-label model.
    with open(MODEL_PATH / "config.json", "r", encoding="utf-8") as file:
        config = json.load(file)

    labels = config.get("id2label", {})
    normalized = {int(key): str(value).lower() for key, value in labels.items()}
    if config.get("problem_type") != "multi_label_classification":
        raise RuntimeError(
            "Expected a multi-label classifier; config problem_type is "
            + repr(config.get("problem_type"))
        )
    if normalized != EXPECTED_LABELS:
        raise RuntimeError(
            f"Unexpected model labels: {normalized}. Expected: {EXPECTED_LABELS}"
        )


def load_model() -> None:
    global tokenizer, model

    logger.info("Preparing model: %s", MODEL_NAME)
    ensure_model_files()

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )
    model.eval()

    if model.config.num_labels != 5:
        raise RuntimeError(
            f"Expected 5 output labels for RuToxic, got {model.config.num_labels}"
        )

    logger.info("RuToxic model loaded successfully")
    logger.info("Labels: %s", model.config.id2label)
    logger.info("Parameters: %s", sum(parameter.numel() for parameter in model.parameters()))


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_text(text: str) -> dict:
    started = time.perf_counter()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS,
        padding=False,
    )

    with torch.inference_mode():
        outputs = model(**inputs)
        probabilities = torch.sigmoid(outputs.logits)[0]

    # Model labels:
    # 0 non-toxic, 1 insult, 2 obscenity, 3 threat, 4 dangerous
    scores = {
        "non_toxic": float(probabilities[0].item()),
        "insult": float(probabilities[1].item()),
        "obscenity": float(probabilities[2].item()),
        "threat": float(probabilities[3].item()),
        "dangerous": float(probabilities[4].item()),
    }

    # Obscenity is intentionally excluded: swearing alone is not a violation.
    violation_scores = {
        "insult": scores["insult"],
        "threat": scores["threat"],
        "dangerous": scores["dangerous"],
    }

    triggered = []
    if scores["insult"] >= INSULT_THRESHOLD:
        triggered.append("insult")
    if scores["threat"] >= THREAT_THRESHOLD:
        triggered.append("threat")
    if scores["dangerous"] >= DANGEROUS_THRESHOLD:
        triggered.append("dangerous")

    toxic_score = max(violation_scores.values())
    is_toxic = bool(triggered)

    latency_ms = (time.perf_counter() - started) * 1000

    return {
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "label": "toxic" if is_toxic else "safe",
        "class": 1 if is_toxic else 0,
        "safe_score": round(scores["non_toxic"], 4),
        "toxic_score": round(toxic_score, 4),
        "threshold": round(
            max(INSULT_THRESHOLD, THREAT_THRESHOLD, DANGEROUS_THRESHOLD), 4
        ),
        "categories": {key: round(value, 4) for key, value in scores.items()},
        "triggered_categories": triggered,
        "ignored_for_auto_mute": ["obscenity"],
        "latency_ms": round(latency_ms, 2),
    }


# ============================================================
# SINGLE WORKER / QUEUE
# ============================================================

async def queue_worker() -> None:
    global processed_requests, failed_requests

    loop = asyncio.get_running_loop()
    logger.info("Queue worker started")

    while True:
        job = await request_queue.get()
        try:
            if job is STOP_WORKER:
                return

            if job.future.cancelled():
                continue

            logger.info("Processing request | queue=%s", request_queue.qsize())
            result = await loop.run_in_executor(
                inference_executor,
                classify_text,
                job.text,
            )

            processed_requests += 1
            if not job.future.done():
                job.future.set_result(result)

            logger.info(
                "Completed | inference=%.2f ms | class=%s | categories=%s",
                result["latency_ms"],
                result["label"],
                result["triggered_categories"],
            )

        except Exception as error:
            failed_requests += 1
            logger.exception("Inference error: %s", error)
            if not job.future.done():
                job.future.set_exception(RuntimeError("Model inference failed"))
        finally:
            if job is not STOP_WORKER:
                request_queue.task_done()


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global request_queue, inference_executor, worker_task
    global ready, accepting_requests, model, tokenizer

    if len(API_TOKEN) < 32:
        raise RuntimeError("Environment variable Token must contain at least 32 characters")

    logger.info("Starting RuToxic Moderation API")

    # Download missing model files, then load model once.
    await asyncio.to_thread(load_model)

    request_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
    inference_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="rutoxic",
    )
    worker_task = asyncio.create_task(queue_worker())

    ready = True
    accepting_requests = True
    logger.info("RuToxic API READY")

    try:
        yield
    finally:
        logger.info("Graceful shutdown started")
        ready = False
        accepting_requests = False

        while not request_queue.empty():
            try:
                job = request_queue.get_nowait()
                if job is not STOP_WORKER and not job.future.done():
                    job.future.set_exception(RuntimeError("Server shutting down"))
                request_queue.task_done()
            except asyncio.QueueEmpty:
                break

        await request_queue.put(STOP_WORKER)
        await worker_task

        inference_executor.shutdown(wait=True, cancel_futures=True)
        model = None
        tokenizer = None
        logger.info("RuToxic API stopped")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="RuToxic Moderation API",
    version="4.0.0",
    lifespan=lifespan,
)


class CheckRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH)


@app.get("/")
async def root():
    return {
        "service": "RuToxic Moderation API",
        "model": MODEL_NAME,
        "status": "online" if ready else "starting",
    }


@app.get("/health")
async def health():
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "online" if ready else "starting",
            "model_loaded": model is not None,
            "model": MODEL_NAME,
            "queue_size": request_queue.qsize() if request_queue is not None else 0,
            "queue_limit": MAX_QUEUE_SIZE,
            "processed": processed_requests,
            "failed": failed_requests,
        },
    )


@app.post("/check")
async def check_text(data: CheckRequest, x_api_token: str = Header(default="")):
    if not hmac.compare_digest(x_api_token, API_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid API token")

    if not ready or not accepting_requests:
        raise HTTPException(status_code=503, detail="Server unavailable")

    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    job = Job(
        text=data.text,
        future=future,
        created_at=time.monotonic(),
    )

    try:
        request_queue.put_nowait(job)
    except asyncio.QueueFull:
        raise HTTPException(
            status_code=429,
            detail="Queue full. Retry later.",
            headers={"Retry-After": "3"},
        )

    try:
        return await asyncio.wait_for(
            asyncio.shield(future),
            timeout=MAX_WAIT_SECONDS,
        )
    except asyncio.TimeoutError:
        future.cancel()
        raise HTTPException(status_code=504, detail="Processing timeout")
    except Exception:
        raise HTTPException(status_code=500, detail="Classification failed")
