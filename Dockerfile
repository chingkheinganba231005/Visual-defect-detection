FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements-service.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.15.0 torchvision==0.29.2 \
 && pip install --no-cache-dir -r requirements-service.txt

# bake the pretrained backbone into the image so the container starts offline
RUN python -c "import torchvision; torchvision.models.wide_resnet50_2(weights='IMAGENET1K_V1')"

COPY src/ src/
COPY models/ models/
ENV PYTHONPATH=/app/src DEFECTS_MODELS_DIR=/app/models DEFECTS_THREADS=4

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "defects.service:app", "--host", "0.0.0.0", "--port", "8000"]
