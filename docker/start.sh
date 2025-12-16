#!/bin/bash
set -e

echo "--- Starting Initialization ---"

# 0. Save environment variables for SSH sessions
printenv >> /etc/environment

# 1. Setup SSH Keys (CRITICAL STEP)
# RunPod passes your web-configured public key as the $PUBLIC_KEY env var.
if [ -n "$PUBLIC_KEY" ]; then
  echo "Setting up SSH access..."
  mkdir -p /root/.ssh
  echo "$PUBLIC_KEY" >/root/.ssh/authorized_keys
  chmod 700 /root/.ssh
  chmod 600 /root/.ssh/authorized_keys
  echo "SSH Key injected successfully."
else
  echo "WARNING: No PUBLIC_KEY environment variable found. SSH access might fail."
fi

# 2. Setup SSH Service
# Ensure the directory exists for privilege separation
mkdir -p /run/sshd
# Start SSH in the background
/usr/sbin/sshd

# 3. Start MLflow UI
# We point the backend store to /workspace so your experiment history
# is saved on the Persistent Volume.
echo "Starting MLflow UI..."
nohup uv run mlflow ui \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:////workspace/mlflow.db \
  --default-artifact-root /workspace/mlruns \
  >/app/mlflow.log 2>&1 &

# 4. Keep the container alive
echo "Container started. Waiting..."
sleep infinity
