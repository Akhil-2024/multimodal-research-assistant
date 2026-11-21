FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/runtime_data

# FAISS / PyTorch CPU runtime dependency
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/runtime_data

EXPOSE 8000
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
