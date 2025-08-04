# Dockerfile

FROM python:3.10

WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port and run FastAPI
CMD ["uvicorn", "src.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]
