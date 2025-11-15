VLA models trained on the data collected from semi autonomous operation to teleoperation.

## Base models

* https://github.com/Physical-Intelligence/openpi
* https://openvla.github.io/


# Reinforcement Learning in Simulation
https://github.com/WiktorJ/sim-robotic-arm-rl/

# Instructions
Run finetuning:
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run aera/autonomous/openpi/scripts/train.py pi0_fast_ar4_mk3_low_mem_finetune --exp-name=test --overwrite`
