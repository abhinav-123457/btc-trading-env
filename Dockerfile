# ──────────────────────────────────────────────────────────────────────────────
# BTC Trading OpenEnv — Dockerfile
# Builds a production-ready FastAPI container for Hugging Face Spaces.
#
# HF Spaces exposes port 7860 by default.
# Build:  docker build -t btc-trading-env .
# Run:    docker run -p 7860:7860 btc-trading-env
# Health: curl http://localhost:7860/health
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# ── Create non-root user (HF Spaces best practice) ───────────────────────────
RUN useradd -m -u 1000 appuser

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies first (layer cache friendly) ─────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY server/ ./server/
COPY models.py environment.py openenv.yaml ./

# ── Ownership ─────────────────────────────────────────────────────────────────
RUN chown -R appuser:appuser /app
USER appuser

# ── Expose HF Spaces port ─────────────────────────────────────────────────────
EXPOSE 7860

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# ── Start server ──────────────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860", \
     "--workers", "1", "--log-level", "info"]
