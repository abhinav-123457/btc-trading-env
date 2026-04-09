FROM python:3.11-slim AS base

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy server package (app.py + ui.html must be together in /app/server/)
COPY server/ ./server/

# Core environment files
COPY models.py environment.py openenv.yaml ./
COPY inference.py ./
COPY __init__.py ./
COPY btc_prices.csv ./

RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# PYTHONPATH=/app so server/app.py can import environment.py and models.py
ENV PYTHONPATH=/app
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", \
     "--workers", "1", "--log-level", "info"]