from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_config = PathJoinSubstitution(
        [FindPackageShare('umi_ros2'), 'config', 'umi_grp.yaml']
    )
    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Path to the parameter file for umi_gripper.',
    )

    node = Node(
        package='umi_ros2',
        executable='umi_gripper_pico',
        name='umi_gripper',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
    )

    return LaunchDescription([config_arg, node])
