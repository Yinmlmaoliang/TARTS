#!/usr/bin/env python3
"""
TARTS Segmentation Launch File

Launches the TARTS segmentation node with parameters loaded from config file.
All parameters are configured via the YAML config file (tarts_params.yaml).

Usage:
    # Use default config file
    ros2 launch tarts_ros tarts_segmentation.launch.py

    # Use custom config file
    ros2 launch tarts_ros tarts_segmentation.launch.py \
        config_file:=/path/to/custom_params.yaml

Note:
    To change parameters (class_name, threshold_method, device, etc.),
    edit the YAML config file directly for consistency.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import UnlessCondition
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for TARTS segmentation node."""

    # Get default config file path
    pkg_share = get_package_share_directory('tarts_ros')
    default_config_file = os.path.join(pkg_share, 'config', 'tarts_params.yaml')

    # Declare launch argument for config file
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_file,
        description='Path to YAML config file for TARTS parameters'
    )

    # Get launch configurations
    config_file = LaunchConfiguration('config_file')

    # Declare launch argument for debug flag
    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug output (true/false)'
    )

    # Get debug configuration
    debug = LaunchConfiguration('debug')

    # TARTS segmentation node - loads all parameters from config file
    tarts_node = Node(
        package='tarts_ros',
        executable='tarts_segmentation',
        name='tarts_segmentation',
        output='screen',
        parameters=[config_file, {'debug': debug}],
        remappings=[
            # Add any topic remappings here if needed
        ]
    )

    return LaunchDescription([
        config_file_arg,
        debug_arg,
        tarts_node,
    ])
