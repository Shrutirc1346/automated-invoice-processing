#=============================
# Step 52 : Build Docker Deployment Environment
#=============================

FROM python:3.11-slim

WORKDIR /app

# Install Tesseract OCR because it is an operating-system package
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Render's default web-service port
EXPOSE 10000

# Start Streamlit
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-10000}"]