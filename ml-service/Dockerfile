# syntax=docker/dockerfile:1
# Aletheia ML service: PaddleOCR + sentence-transformers behind HTTP.
# This image is intentionally large. Backend code changes must NOT
# invalidate these layers: requirements are copied before app code,
# and model pre-warming runs before the app copy.
FROM python:3.12-slim

# System libraries required by PaddlePaddle/OpenCV at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ml-service/requirements.txt ./requirements.txt
# BuildKit cache mount persists pip's download cache across builds. The
# cache lives outside the image layer and is never baked into it.
RUN --mount=type=cache,id=ml-pip-cache,target=/root/.cache/pip \
    pip install --default-timeout=1800 --retries=10 -r requirements.txt

# Bake model weights into the image so the running container never
# downloads models on first request.
ENV PADDLEX_HOME=/root/.paddlex
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en')"

ENV HF_HUB_CACHE=/root/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY ml-service/app ./app

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
