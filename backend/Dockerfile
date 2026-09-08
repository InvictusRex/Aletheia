FROM python:3.12-slim

# System libraries required by PaddlePaddle/OpenCV at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-warm PaddleOCR model weights at build time so the running
# container never downloads models (unit tests never initialize OCR).
ENV PADDLEX_HOME=/root/.paddlex
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en')"

COPY backend/app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
