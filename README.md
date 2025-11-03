# AERA

## Autonomous Experimental Robotic Arm

Building robotic arm controlled by natural language.

[![YouTube](http://i.ytimg.com/vi/ndCuwOP6PO8/hqdefault.jpg)](https://www.youtube.com/watch?v=ndCuwOP6PO8)

Using teleoperation and semi-autonomous, zero-shot methods to collect training data, and leveraging pre-trained
visual-linguistic alignment (VLA) models to create a framework for training robotic arms to perform various actions in
dynamic environments.

Fully opensource software and hardware.

## Autonomous

RL and VLA models to convert vision and language into actions.

## Semi Autonomous

Using object detection (e.g. [Grounding Dino](https://huggingface.co/docs/transformers/en/model_doc/grounding-dino))
and image segmentation (e.g. [Segment Anything](https://github.com/facebookresearch/segment-anything)) SOTA zero-shot
models to prepare input to algorithmic manipulation methods.

## Teleoperation

WIP

## Relevant Repos

https://github.com/wiktorj/easy_handeye2

https://github.com/wiktorj/ros2_aruco.git

https://github.com/ycheng517/pymoveit2.git

https://github.com/wiktorj/ar4_ros_driver.git

https://github.com/wiktorj/sim-robotic-arm-rl/

# Credits

https://www.anninrobotics.com/ - For arm design

https://github.com/ycheng517 - for AR4 MK3 arm drivers and calibration software


# Installation tips:

When using mamba run the following to install ros dependencies:

`mamba install -c conda-forge compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep`

`mamba install -c robostack-jazzy -c conda-forge ros-jazzy-geometry-msgs ros-jazzy-sensor-msgs open3d glfw  numpy scipy opencv`

`pip install -e .`

In semi-autonomous:
`pip install -r requirements.txt`
`pip install --no-build-isolation -e GroundingDINO`

In semi-autonomous/aera_semi_autonomous:
`pip install -e . --use-pep517`

## Env for working with models
Because of package conflicts, it is recommended to create a new environment for working with models.

`uv venv`
`source .venv/bin/activate`

In different directory
`git clone --recurse-submodules git@github.com:Physical-Intelligence/openpi.git`
`GIT_LFS_SKIP_SMUDGE=1 uv sync`
`GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .`
`uv pip install --reinstall 'lerobot[all]'`
