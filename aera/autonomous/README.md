VLA models trained on the data collected from semi autonomous operation to teleoperation.

## Base models

* https://github.com/Physical-Intelligence/openpi
* https://openvla.github.io/


# Reinforcement Learning in Simulation
https://github.com/WiktorJ/sim-robotic-arm-rl/

# Instructions
Run finetuning:
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run aera/autonomous/openpi/scripts/train.py pi0_fast_ar4_mk3_low_mem_finetune --base-config.exp-name=test --base-config.overwrite`


# Docker instructions
build image for RunPod: `docker build -t wiktorj/aera-runpod:1 -f docker/Dockerfile.runpod .`

run with ssh and storage binding:
```

docker run --rm -d \
    --gpus all \
    --name aera-local \
    -p 2222:22 \
    -p 5000:5000 \
    -e PUBLIC_KEY="$(cat ~/.ssh/id_ed25519.pub)" \
    -v $(pwd)/runpod_volume:/workspace \
    wiktorj/aera-runpod:1
```

Connect to ssh server: 

```
ssh -p 2222 root@localhost -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
```
```
