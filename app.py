import asyncio
import hmac
import json
import logging
import os
import time
import urllib.error
import urllib.request
import gc
import io

from PIL import Image
from fastapi import UploadFile, File

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

IMAGE_MODEL_PATH = MODEL_PATH / "image_model"
IMAGE_MODEL_CONFIG_URL = "https://huggingface.co/OwenElliott/image-safety-classifier-xs/resolve/main/config.json?download=true"
IMAGE_MODEL_WEIGHTS_URL = "https://huggingface.co/OwenElliott/image-safety-classifier-xs/resolve/main/model.safetensors?download=true"
IMAGE_LABELS = ["NSFL", "NSFW", "SFW"]

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
image_model = None
image_model_loaded = False
image_requests = 0
image_failures = 0

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
# LAZY IMAGE CLASSIFIER
# ============================================================

def ensure_image_model_files() -> None:
    IMAGE_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    download_file_to(IMAGE_MODEL_PATH / "config.json", IMAGE_MODEL_CONFIG_URL)
    download_file_to(IMAGE_MODEL_PATH / "model.safetensors", IMAGE_MODEL_WEIGHTS_URL)
    with open(IMAGE_MODEL_PATH / "config.json", "r", encoding="utf-8") as file:
        config = json.load(file)
    if config.get("architecture") != "swiftformer_xs" or config.get("num_classes") != 3:
        raise RuntimeError(f"Unexpected image model config: {config}")
    names = config.get("pretrained_cfg", {}).get("label_names")
    if names and names != IMAGE_LABELS:
        raise RuntimeError(f"Unexpected image labels: {names}")


def download_file_to(target: Path, url: str) -> None:
    if target.is_file() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    last_error = None
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            logger.info("Downloading image model file %s (attempt %s/%s)", target.name, attempt, DOWNLOAD_RETRIES)
            request = urllib.request.Request(url, headers={"User-Agent": "Ksenya-Image-Moderation/1.0"})
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, open(temporary, "wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if not temporary.exists() or temporary.stat().st_size == 0:
                raise RuntimeError(f"Image model file is empty: {target.name}")
            temporary.replace(target)
            logger.info("Downloaded image model file %s (%s bytes)", target.name, target.stat().st_size)
            return
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(min(attempt * 2, 5))
    raise RuntimeError(f"Could not download image model file {target.name}: {last_error}")


def _image_tensor(raw: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    # config.json: 224x224, bicubic, crop_pct=0.95, center crop, ImageNet stats.
    resize_size = int(round(224 / 0.95))
    width, height = image.size
    scale = resize_size / min(width, height)
    resized = (max(resize_size, int(round(width * scale))), max(resize_size, int(round(height * scale))))
    image = image.resize(resized, resample=Image.Resampling.BICUBIC)
    left = (image.width - 224) // 2
    top = (image.height - 224) // 2
    image = image.crop((left, top, left + 224, top + 224))
    import numpy as np
    array = np.asarray(image, dtype="float32") / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def classify_images_cached(raw_images: list[bytes]) -> dict:
    """Загружает SwiftFormer XS один раз и держит его в памяти между запросами.

    Это убирает повторное создание модели и чтение весов для каждого фото.
    Inference выполняется в общем однопоточном executor, поэтому одновременной
    загрузки второй копии модели не возникает.
    """
    global image_model, image_model_loaded, image_requests, image_failures
    import timm
    from timm.models.swiftformer import checkpoint_filter_fn
    from safetensors.torch import load_file

    started = time.perf_counter()
    try:
        if image_model is None or not image_model_loaded:
            ensure_image_model_files()
            loaded_model = timm.create_model("swiftformer_xs", pretrained=False, num_classes=3)
            state = load_file(str(IMAGE_MODEL_PATH / "model.safetensors"), device="cpu")
            if state and all(key.startswith("module.") for key in state):
                state = {key[7:]: value for key, value in state.items()}
            if state and all(key.startswith("model.") for key in state):
                state = {key[6:]: value for key, value in state.items()}
            state = checkpoint_filter_fn(state, loaded_model)
            missing, unexpected = loaded_model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise RuntimeError(
                    f"Image model weights mismatch; missing={missing[:8]}, unexpected={unexpected[:8]}"
                )
            loaded_model.eval()
            image_model = loaded_model
            image_model_loaded = True
            logger.info("Image safety classifier loaded and cached in memory")

        batch = torch.stack([_image_tensor(raw) for raw in raw_images])
        with torch.inference_mode():
            logits = image_model(batch)
            if isinstance(logits, (tuple, list)):
                logits = sum(logits) / len(logits)
            probabilities = torch.softmax(logits, dim=1).cpu().tolist()

        results = []
        violation = False
        for scores in probabilities:
            top_index = max(range(len(scores)), key=lambda index: scores[index])
            label = IMAGE_LABELS[top_index]
            is_unsafe = label in ("NSFL", "NSFW")
            violation = violation or is_unsafe
            results.append({
                "label": label,
                "scores": {name: round(float(scores[index]), 6) for index, name in enumerate(IMAGE_LABELS)},
                "violation": is_unsafe,
            })
        image_requests += 1
        return {
            "model": "OwenElliott/image-safety-classifier-xs",
            "label_names": IMAGE_LABELS,
            "violation": violation,
            "results": results,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    except Exception:
        image_failures += 1
        raise

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
        image_model = None
        image_model_loaded = False
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
            "image_model_loaded": image_model_loaded,
            "image_processed": image_requests,
            "image_failed": image_failures,
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


@app.post("/check-image")
async def check_images(files: list[UploadFile] = File(...), x_api_token: str = Header(default="")):
    if not hmac.compare_digest(x_api_token, API_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid API token")
    if not ready or not accepting_requests:
        raise HTTPException(status_code=503, detail="Server unavailable")
    if not files or len(files) > 5:
        raise HTTPException(status_code=400, detail="Send between 1 and 5 image frames")
    raw_images = []
    for upload in files:
        content = await upload.read()
        if not content or len(content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image is empty or too large")
        raw_images.append(content)
    loop = asyncio.get_running_loop()
    try:
        # Один общий executor (max_workers=1) сериализует текстовые и визуальные inference.
        result = await loop.run_in_executor(inference_executor, classify_images_cached, raw_images)
        return result
    except Exception as error:
        logger.exception("Image classification failed: %s", error)
        raise HTTPException(status_code=500, detail="Image classification failed")
