# Docker Setup for AERA Training

This directory contains Docker configuration for training AERA models using OpenPI.

## Prerequisites

1. **Docker Installation**: Install Docker following the [official instructions](https://docs.docker.com/engine/install/)
2. **Rootless Mode**: Docker must be installed in [rootless mode](https://docs.docker.com/engine/security/rootless/)
3. **NVIDIA Container Toolkit**: Install the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to use GPU acceleration

### Quick Setup for Ubuntu 22.04

If you're starting from scratch on Ubuntu 22.04, you can use the convenience scripts from the OpenPI repository:

```bash
# Install Docker
curl -fsSL https://raw.githubusercontent.com/WiktorJ/openpi/main/scripts/docker/install_docker_ubuntu22.sh | bash

# Install NVIDIA Container Toolkit
curl -fsSL https://raw.githubusercontent.com/WiktorJ/openpi/main/scripts/docker/install_nvidia_container_toolkit.sh | bash
```

## Usage

### 1. Build and Start the Container

```bash
docker compose -f docker/docker-compose.train.yml up --build
```

This will:
- Build the Docker image with all dependencies
- Start the container with GPU access
- Mount the current directory to `/workspace`
- Set up the environment for training

### 2. Run Training

Once inside the container, you can run training with:

```bash
uv run aera/autonomous/openpi/scripts/train.py <config_name> --exp-name=<experiment_name> --overwrite
```

For example:
```bash
uv run aera/autonomous/openpi/scripts/train.py my_config --exp-name=test_experiment --overwrite
```

### 3. Interactive Development

To get an interactive shell in the container:

```bash
docker compose -f docker/docker-compose.train.yml exec aera-train bash
```

Or if the container isn't running yet:

```bash
docker compose -f docker/docker-compose.train.yml run --rm aera-train bash
```

## Environment Variables

The following environment variables are automatically set:

- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`: Limits JAX memory usage to 90% of available GPU memory
- `NVIDIA_VISIBLE_DEVICES=all`: Makes all GPUs available to the container
- `PYTHONUNBUFFERED=1`: Ensures Python output is not buffered

## Volumes

- `.:/workspace`: Mounts the current directory to `/workspace` in the container
- `~/.cache:/root/.cache`: Mounts the host cache directory for JAX compilation cache and model checkpoints

## Troubleshooting

### GPU Not Available

If you get errors about GPU not being available:

1. Ensure NVIDIA drivers are installed on the host
2. Verify NVIDIA Container Toolkit is properly installed
3. Check that `nvidia-smi` works on the host
4. Ensure Docker is not installed via snap (incompatible with NVIDIA runtime)

### Memory Issues

If you encounter out-of-memory errors:

1. Adjust `XLA_PYTHON_CLIENT_MEM_FRACTION` to a lower value (e.g., 0.7)
2. Reduce batch size in your training configuration
3. Use gradient accumulation if supported

### Build Issues

If the Docker build fails:

1. Ensure you have a stable internet connection
2. Try building with `--no-cache` flag: `docker compose -f docker/docker-compose.train.yml build --no-cache`
3. Check that all dependencies in `pyproject.toml` are accessible

## Development Workflow

1. Make changes to your code on the host machine
2. The changes are immediately available in the container due to volume mounting
3. Run training or tests inside the container
4. Results and checkpoints are saved back to the host through volume mounting
