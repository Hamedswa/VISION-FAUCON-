# ─────────────────────────────────────────────────────────────────
#  TRADING BOT — Dockerfile
#  Optimisé pour Railway (multi-stage, image légère)
# ─────────────────────────────────────────────────────────────────

# ── Stage 1 : Build des dépendances ──────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copie et installation des dépendances Python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2 : Image de production ────────────────────────────────
FROM python:3.11-slim AS production

WORKDIR /app

# Dépendances runtime uniquement
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copie des packages Python installés
COPY --from=builder /root/.local /root/.local

# Copie du code source
COPY . .

# Crée les dossiers nécessaires
RUN mkdir -p logs \
    && mkdir -p backtesting/results \
    && mkdir -p backtesting/reports

# Variables d'environnement par défaut
ENV PATH=/root/.local/bin:$PATH \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=UTC

# Port du dashboard
EXPOSE 8000

# Health check — vérifie que le dashboard répond
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Utilisateur non-root pour la sécurité
RUN useradd -m -u 1000 botuser && \
    chown -R botuser:botuser /app
USER botuser

# Point d'entrée
CMD ["python", "main.py"]
