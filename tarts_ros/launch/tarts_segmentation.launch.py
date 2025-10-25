#!/usr/bin/env python3
"""
TARTS Segmentation Launch File

Launches the TARTS segmentation node with configurable parameters.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    """Generate launch description for TARTS segmentation node."""

    # Declare launch arguments
    class_name_arg = DeclareLaunchArgument(
        'class_name',
        default_value='corn',
        description='Name of the object class to segment'
    )

    threshold_arg = DeclareLaunchArgument(
        'threshold',
        default_value='0.5',
        description='Similarity threshold for segmentation'
    )

    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device to use (cuda or cpu)'
    )

    input_size_arg = DeclareLaunchArgument(
        'input_size',
        default_value='480',
        description='Input image size'
    )

    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/camera/color/image_raw',
        description='Input image topic'
    )

    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='false',
        description='Enable debug output'
    )

    # TARTS segmentation node
    tarts_node = Node(
        package='tarts_ros',
        executable='tarts_segmentation',
        name='tarts_segmentation',
        output='screen',
        parameters=[{
            'class_name': LaunchConfiguration('class_name'),
            'threshold': LaunchConfiguration('threshold'),
            'device': LaunchConfiguration('device'),
            'input_size': LaunchConfiguration('input_size'),
            'image_topic': LaunchConfiguration('image_topic'),
            'debug': LaunchConfiguration('debug'),
        }],
        remappings=[
            # Add any topic remappings here if needed
        ]
    )

    return LaunchDescription([
        class_name_arg,
        threshold_arg,
        device_arg,
        input_size_arg,
        image_topic_arg,
        debug_arg,
        tarts_node,
    ])
