from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction, RegisterEventHandler, LogInfo, OpaqueFunction, \
    DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, FindExecutable, \
    Command, TextSubstitution, PythonExpression
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterFile
import os
import yaml


def generate_launch_description():
    serial_port = LaunchConfiguration("serial_port")
    calibrate = LaunchConfiguration("calibrate")
    include_gripper = LaunchConfiguration("include_gripper")
    arduino_serial_port = LaunchConfiguration("arduino_serial_port")
    ar_model_config = LaunchConfiguration("ar_model")
    log_level = LaunchConfiguration("log_level")

    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name="xacro")]),
        " ",
        PathJoinSubstitution([
            FindPackageShare("annin_ar4_driver"), "urdf", "ar.urdf.xacro"
        ]),
        " ",
        "ar_model:=",
        ar_model_config,
        " ",
        "serial_port:=",
        serial_port,
        " ",
        "calibrate:=",
        calibrate,
        " ",
        "include_gripper:=",
        include_gripper,
        " ",
        "arduino_serial_port:=",
        arduino_serial_port,
        " ",
        "log_level:=",
        log_level
    ])
    robot_description = {"robot_description": robot_description_content}

    joint_controllers_cfg = PathJoinSubstitution([
        FindPackageShare("annin_ar4_driver"), "config", "controllers.yaml"
    ])

    update_rate_config_file = PathJoinSubstitution([
        FindPackageShare("annin_ar4_driver"),
        "config",
        "controller_update_rate.yaml",
    ])

    controller_manager_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            update_rate_config_file,
        ],
        remappings=[('~/robot_description', 'robot_description')],
        output="screen",
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )

    # --- Controller Spawning and Activation Logic (inside OpaqueFunction) ---
    def controller_spawning_and_activation(context, *args, **kwargs):
        # 1. Resolve Paths
        resolved_joint_controllers_cfg = context.perform_substitution(joint_controllers_cfg)

        # 2. Create temporary parameter files
        def create_temp_param_file(controller_name, controller_type, config_file):
            temp_param_file = os.path.join('/tmp', f'{controller_name}_params.yaml')
            with open(config_file, 'r') as f:
                controller_config = yaml.safe_load(f)

            controller_params = {controller_name: {
                'ros__parameters': controller_config.get(controller_name, {}).get('ros__parameters', {})}}
            controller_params[controller_name]['ros__parameters']['type'] = controller_type
            with open(temp_param_file, 'w') as f:
                yaml.dump(controller_params, f)
            return temp_param_file

        joint_traj_param_file = create_temp_param_file("joint_trajectory_controller",
                                                       "joint_trajectory_controller/JointTrajectoryController",
                                                       resolved_joint_controllers_cfg)
        gripper_param_file = create_temp_param_file("gripper_controller",
                                                    "position_controllers/GripperActionController",
                                                    resolved_joint_controllers_cfg) if context.perform_substitution(
            include_gripper) == 'True' else None
        joint_broadcaster_param_file = create_temp_param_file("joint_state_broadcaster",
                                                              "joint_state_broadcaster/JointStateBroadcaster",
                                                              resolved_joint_controllers_cfg)

        # 3. Create Spawner Nodes
        spawner_nodes = []

        spawner_nodes.append(Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "joint_trajectory_controller",
                "-c",
                "/controller_manager",
                "--controller-manager-timeout",
                "60",
                "--param-file", joint_traj_param_file,
                "--inactive",  # Start inactive!
                '--ros-args', '--log-level',
                context.perform_substitution(log_level)  # Resolve log_level
            ],
        ))

        if gripper_param_file:
            spawner_nodes.append(Node(
                package="controller_manager",
                executable="spawner",
                arguments=[
                    "gripper_controller",
                    "-c",
                    "/controller_manager",
                    "--controller-manager-timeout",
                    "60",
                    "--param-file", gripper_param_file,
                    "--inactive",  # Start inactive!
                    '--ros-args', '--log-level',
                    context.perform_substitution(log_level)  # Resolve log_level
                ],
            ))

        spawner_nodes.append(Node(
            package="controller_manager",
            executable="spawner",
            arguments=[
                "joint_state_broadcaster",
                "-c",
                "/controller_manager",
                "--controller-manager-timeout",
                "60",
                "--param-file", joint_broadcaster_param_file,
                "--inactive",  # Start inactive!
                '--ros-args', '--log-level',
                context.perform_substitution(log_level)  # Resolve log_level
            ],
        ))

        # 4. Activate controllers AFTER a delay using ros2 control commands
        delay_seconds = 60.0

        commands = []
        # Activate controllers (CORRECTED SYNTAX)
        commands.append("ros2 control set_controller_state joint_trajectory_controller active")
        if context.perform_substitution(include_gripper) == 'True':
            commands.append("ros2 control set_controller_state gripper_controller active")
        commands.append("ros2 control set_controller_state joint_state_broadcaster active")

        delayed_activation = TimerAction(
            period=delay_seconds,
            actions=[
                ExecuteProcess(
                    cmd=[[cmd]],
                    shell=True
                ) for cmd in commands
            ]
        )

        return [*spawner_nodes, delayed_activation]

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')]
    )

    ld = LaunchDescription()

    # Launch Arguments
    ld.add_action(
        DeclareLaunchArgument(
            "serial_port",
            default_value="/dev/ttyACM0",
            description="Serial port to connect to the robot",
        ))
    ld.add_action(
        DeclareLaunchArgument(
            "calibrate",
            default_value="True",
            description="Calibrate the robot on startup",
            choices=["True", "False"],
        ))
    ld.add_action(
        DeclareLaunchArgument(
            "include_gripper",
            default_value="True",
            description="Run the servo gripper",
            choices=["True", "False"],
        ))
    ld.add_action(
        DeclareLaunchArgument(
            "arduino_serial_port",
            default_value="/dev/ttyUSB0",
            description="Serial port of the Arduino nano for the servo gripper",
        ))
    ld.add_action(
        DeclareLaunchArgument("ar_model",
                              default_value="mk3",
                              choices=["mk1", "mk2", "mk3"],
                              description="Model of AR4"))
    ld.add_action(DeclareLaunchArgument(
        "log_level",
        default_value=TextSubstitution(text=str("INFO")),
        description="Logging level"
    ))

    ld.add_action(controller_manager_node)
    ld.add_action(robot_state_publisher_node)
    ld.add_action(OpaqueFunction(function=controller_spawning_and_activation))
    return ld