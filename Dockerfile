# Start from an official, small Python image (a supported Python version).
FROM python:3.12-slim

# All following commands run inside /app in the container.
WORKDIR /app

# Install dependencies first. Docker caches this layer, so rebuilds are fast
# unless requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the files the service needs (the tests are not shipped in the image).
COPY app.py model.py ./

# Listen on all interfaces so the port can be published outside the container.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]
