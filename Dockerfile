FROM python:3.10-slim-bookworm

# noninteractive installs
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Copy code
COPY . /app

# Install system dependencies
# - awscli: for S3 / cloud operations 
# - ffmpeg, libsndfile1, libstdc++6: required for librosa / soundfile / opensmile / torchaudio
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        awscli \
        ffmpeg \
        libsndfile1 \
        libstdc++6 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8080

# Run FastAPI app
CMD ["python3", "app.py"]