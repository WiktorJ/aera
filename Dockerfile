FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

# Install JAX with CUDA support and the project dependencies
RUN pip install --upgrade pip && \
    pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
    pip install .

ENTRYPOINT ["python3"]
CMD ["aera/autonomous/openpi/scripts/train.py", "--help"]
