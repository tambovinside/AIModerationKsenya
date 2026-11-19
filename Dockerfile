
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MALLOC_ARENA_MAX=2

# CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    --index-url https://download.pytorch.org/whl/cpu

# API dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Application
COPY app.py .
COPY values.py .

# Local model
COPY model ./model

EXPOSE 10000

# Single process, bounded HTTP concurrency
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --limit-concurrency 20 --backlog 32 --timeout-keep-alive 5 --no-access-log"]
