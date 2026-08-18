FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY shared /app/shared
COPY services/llm_service /app/services/llm_service

EXPOSE 8005
CMD ["uvicorn", "services.llm_service.app:app", "--host", "0.0.0.0", "--port", "8005"]
