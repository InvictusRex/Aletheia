# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /code

COPY backend/requirements.txt ./requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --default-timeout=1800 --retries=10 -r requirements.txt

COPY backend/alembic.ini ./alembic.ini
COPY backend/alembic ./alembic

COPY backend/app ./app

EXPOSE 8000

# Migrations run before the API accepts traffic: Compose waits for the
# database healthcheck, so `docker compose up` brings the schema to head
# on a fresh volume and is a no-op on an existing one.
CMD ["sh", "-c", "alembic -c alembic.ini upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port 8000"]
