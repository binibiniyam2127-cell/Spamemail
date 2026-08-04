FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "deployment.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]
