FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/inference/ ./inference/
COPY models/ ./models/  # Copy best model
CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]