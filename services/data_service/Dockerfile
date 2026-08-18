FROM python:3.11-slim

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY shared /app/shared
COPY fusion /app/fusion
COPY langgraph /app/langgraph
COPY services/data_service /app/services/data_service
COPY data /app/data

EXPOSE 8001
CMD ["uvicorn", "services.data_service.app:app", "--host", "0.0.0.0", "--port", "8001"]
