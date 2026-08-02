# Minimal ROS message stubs

`ar4_mk3_robot_interface` and `pick_and_place_helpers` import `geometry_msgs` /
`sensor_msgs` / `std_msgs` / `cv_bridge` at module scope, so the scripted expert
cannot be driven from a plain Python venv — only from inside the ROS container.
Every measurement of the demonstrated arm (timing, per-step deltas, grasp
physics) needs exactly that code path and nothing else from ROS.

These stubs supply just the attribute-bag message classes those modules touch.
`measure_scripted_arm.py` puts this directory on `sys.path` **only when the real
packages are missing**, so inside the container the genuine messages still win.

Not a substitute for ROS: `CvBridge` raises on use. That is deliberate — the
image path is only reached when a `TrajectoryDataCollector` is attached, which
the measurement tools never do. If you hit that error, you asked a headless tool
to record a dataset; use `collect_trajectories.py` in the container instead.
