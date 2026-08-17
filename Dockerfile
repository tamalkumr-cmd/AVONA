FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1. Install lightweight CPU PyTorch directly from official wheels
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of the dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY unified_server.py .
COPY my_finetuned_slm/ ./my_finetuned_slm/

EXPOSE 8000

CMD ["uvicorn", "unified_server:app", "--host", "0.0.0.0", "--port", "8000"]