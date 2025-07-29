FROM python:3.10-slim

WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-ita \
    tesseract-ocr-eng \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgfortran5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Verify tesseract languages
RUN tesseract --list-langs


# Set environment variables for headless operation
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV OPENCV_IO_ENABLE_JASPER=1


# Copy application code
COPY . .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt


# Create non-root user
#RUN adduser --disabled-password --gecos "" appuser && \
#    chown -R appuser:appuser /app

#USER appuser

# Expose port
EXPOSE 9292

# Healthcheck
#HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
#  CMD curl -f http://localhost:9292/health || exit 1

# Start command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9292", "--reload"]
