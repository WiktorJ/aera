#!/bin/bash
set -e

# 1. Start setup SSH
# Ensure the directory exists for privilege separation
mkdir -p /run/sshd

# Start SSH in the background
echo "Starting SSH..."
/usr/sbin/sshd

# 2. Start MLflow UI
# We point the backend store to /workspace so your experiment history
# is saved on the Persistent Volume, not the ephemeral container.
echo "Starting MLflow UI..."
# Use nohup to run it in the background and log output
nohup uv run mlflow ui \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:////workspace/mlflow.db \
  --default-artifact-root /workspace/mlruns \
  >/app/mlflow.log 2>&1 &

# 3. Keep the container alive
# This prevents the container from exiting so you can SSH in.
echo "Container started. Waiting..."
sleep infinity
