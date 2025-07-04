import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.substitutions import TextSubstitution


def load_yaml(package_name, file_name):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_name)
    with open(absolute_file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def generate_launch_description():
    log_level = LaunchConfiguration("log_level")
    custom_camera_params_file = os.path.join(
        get_package_share_directory("aera_semi_autonomous"), "config", "camera.yaml"
    )

    launch_args = {
        "rs_compat": "true",
        "pointcloud.enable": "true",
        "params_file": custom_camera_params_file,
    }
    depthai = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("depthai_ros_driver"),
                    "launch",
                    "camera.launch.py",
                )
            ]
        ),
        launch_arguments=launch_args.items(),
    )

    calibration_tf_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("easy_handeye2"),
                    "launch",
                    "publish.launch.py",
                )
            ]
        ),
        launch_arguments={"name": "ar4_calibration"}.items(),
    )

    delay_calibration_tf_publisher = TimerAction(
        actions=[calibration_tf_publisher],
        period=2.0,
    )

    ar_moveit_launch = PythonLaunchDescriptionSource(
        [
            os.path.join(
                get_package_share_directory("annin_ar4_moveit_config"),
                "launch",
                "moveit.launch.py",
            )
        ]
    )
    ar_moveit_args = {
        "include_gripper": "True",
        "rviz_config_file": "moveit_with_camera.rviz",
    }.items()
    ar_moveit = IncludeLaunchDescription(
        ar_moveit_launch, launch_arguments=ar_moveit_args
    )

    aera_semi_autonomous_node = Node(
        package="aera_semi_autonomous",
        executable="aera_semi_autonomous_node",
        name="aera_semi_autonomous_node",
        output="screen",
        arguments=["--ros-args", "--log-level", LaunchConfiguration("log_level")],
    )

    static_tf_publisher = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=["0", "0", "0", "0", "0", "0", "world", "camera_link"],
        output="screen",
    )

    log_evel_arg = DeclareLaunchArgument(
        "log_level",
        default_value=TextSubstitution(text=str("INFO")),
        description="Logging level",
    )

    return LaunchDescription(
        [
            depthai,
            delay_calibration_tf_publisher,
            ar_moveit,
            aera_semi_autonomous_node,
            static_tf_publisher,
            log_evel_arg,
        ]
    )
