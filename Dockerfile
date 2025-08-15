FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for common Python libs (psycopg2/psycopg, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY test.txt .
RUN python -m pip install --upgrade pip && pip install -r test.txt
# Copy backend code
COPY . .

# Expose API port
EXPOSE 8000

# Start FastAPI via uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
