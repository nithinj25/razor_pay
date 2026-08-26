# One image, three entrypoints (ingress, worker, console). They share a
# codebase and differ only in the process they start, so building three
# images would triple the layer cache for no benefit.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first: application code changes far more often than
# requirements.txt, so this layer stays cached across most rebuilds.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY core/      ./core/
COPY services/  ./services/
COPY harness/   ./harness/
COPY web/       ./web/
COPY analytics/ ./analytics/
COPY pytest.ini README.md ./

# Never run as root: the container has network access to the payment
# gateway, so a compromise here should not also be a container escape.
RUN useradd --create-home --uid 10001 nishchay && chown -R nishchay:nishchay /app
USER nishchay

EXPOSE 8000 8080
CMD ["uvicorn", "services.ingress.main:app", "--host", "0.0.0.0", "--port", "8000"]
