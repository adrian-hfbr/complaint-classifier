# Dockerfile

# --- STAGE 1: The "Builder" Stage ---
# This stage installs dependencies into a temporary image.
FROM python:3.11-slim as builder

WORKDIR /app

COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# --- STAGE 2: The "Production" Stage ---
# This is the final, clean image.
FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

COPY ./src ./src
COPY ./models ./models

# Expose the port the app runs on.
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
