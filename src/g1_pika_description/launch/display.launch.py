from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_share = Path(get_package_share_directory("g1_pika_description"))
    default_model = package_share / "urdf" / "g1_29dof_pika.urdf"
    default_rviz = package_share / "rviz" / "display.rviz"

    model = LaunchConfiguration("model")
    use_gui = LaunchConfiguration("use_gui")
    use_rviz = LaunchConfiguration("use_rviz")
    base_x = LaunchConfiguration("base_x")
    base_y = LaunchConfiguration("base_y")
    base_z = LaunchConfiguration("base_z")
    base_roll = LaunchConfiguration("base_roll")
    base_pitch = LaunchConfiguration("base_pitch")
    base_yaw = LaunchConfiguration("base_yaw")

    robot_description = ParameterValue(Command(["cat ", model]), value_type=str)

    return LaunchDescription(
        [
            DeclareLaunchArgument("model", default_value=str(default_model)),
            DeclareLaunchArgument("use_gui", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("base_x", default_value="0.0"),
            DeclareLaunchArgument("base_y", default_value="0.0"),
            DeclareLaunchArgument("base_z", default_value="0.0"),
            DeclareLaunchArgument("base_roll", default_value="0.0"),
            DeclareLaunchArgument("base_pitch", default_value="0.0"),
            DeclareLaunchArgument("base_yaw", default_value="0.0"),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="base_link_to_pelvis",
                arguments=[
                    "--x",
                    base_x,
                    "--y",
                    base_y,
                    "--z",
                    base_z,
                    "--roll",
                    base_roll,
                    "--pitch",
                    base_pitch,
                    "--yaw",
                    base_yaw,
                    "--frame-id",
                    "base_link",
                    "--child-frame-id",
                    "pelvis",
                ],
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                condition=IfCondition(use_gui),
            ),
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                condition=UnlessCondition(use_gui),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", str(default_rviz)],
                condition=IfCondition(use_rviz),
                output="screen",
            ),
        ]
    )
