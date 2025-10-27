#!/usr/bin/env python3
"""
TARTS Online Update Launch File

Launches both segmentation node and prototype update node for online adaptation.
All parameters are loaded from the config file (online_update_params.yaml).

Usage:
    # Use default config file
    ros2 launch tarts_ros tarts_online_update.launch.py

    # Use custom config file
    ros2 launch tarts_ros tarts_online_update.launch.py \
        config_file:=/path/to/custom_online_update_params.yaml

Note:
    To change parameters, edit the YAML config file directly.
    This ensures consistency between segmentation and prototype update nodes.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Generate launch description for TARTS online update system."""

    # Get default config file path
    pkg_share = get_package_share_directory('tarts_ros')
    default_config_file = os.path.join(pkg_share, 'config', 'online_update_params.yaml')

    # Declare launch argument for config file
    config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config_file,
        description='Path to YAML config file for TARTS online update parameters'
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

    # Segmentation node - loads all parameters from config file
    segmentation_node = Node(
        package='tarts_ros',
        executable='tarts_segmentation',
        name='tarts_segmentation',
        output='screen',
        parameters=[config_file, {'debug': debug}],
        remappings=[
            # Add any topic remappings here if needed
        ]
    )

    # Prototype update node - loads all parameters from config file
    prototype_update_node = Node(
        package='tarts_ros',
        executable='tarts_prototype_update',
        name='tarts_prototype_update',
        output='screen',
        parameters=[config_file, {'debug': debug}]
    )

    return LaunchDescription([
        # Declare arguments
        config_file_arg,
        debug_arg,

        # Launch nodes
        segmentation_node,
        prototype_update_node,
    ])
