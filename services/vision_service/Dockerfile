FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY shared /app/shared
COPY fusion /app/fusion
COPY models /app/models
COPY data /app/data
COPY services/vision_service /app/services/vision_service

EXPOSE 8004
CMD ["uvicorn", "services.vision_service.app:app", "--host", "0.0.0.0", "--port", "8004"]
