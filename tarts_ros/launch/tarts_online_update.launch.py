#!/usr/bin/env python3
"""
TARTS Online Update Launch File

Launches both segmentation node and prototype update node for online adaptation.

Usage:
    ros2 launch tarts_ros tarts_online_update.launch.py \
        class_name:=corn \
        threshold:=0.5 \
        device:=cuda
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    """Generate launch description for TARTS online update system."""

    # Declare launch arguments
    class_name_arg = DeclareLaunchArgument(
        'class_name',
        default_value='corn',
        description='Object class name to segment'
    )

    threshold_arg = DeclareLaunchArgument(
        'threshold',
        default_value='0.5',
        description='Similarity threshold for binary mask generation [0.0-1.0]'
    )

    device_arg = DeclareLaunchArgument(
        'device',
        default_value='cuda',
        description='Device to use (cuda or cpu)'
    )

    input_size_arg = DeclareLaunchArgument(
        'input_size',
        default_value='480',
        description='Input image size for processing'
    )

    prototype_dir_arg = DeclareLaunchArgument(
        'prototype_dir',
        default_value=os.path.expanduser('~/.ros/tarts_prototypes'),
        description='Directory containing prototype files'
    )

    momentum_arg = DeclareLaunchArgument(
        'momentum',
        default_value='0.9',
        description='Momentum coefficient for prototype update [0.0-1.0]'
    )

    update_interval_arg = DeclareLaunchArgument(
        'update_interval',
        default_value='0.5',
        description='Distance interval for prototype updates (meters)'
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

    # Prototype Update Node selection arguments
    min_observation_distance_arg = DeclareLaunchArgument(
        'min_observation_distance',
        default_value='1.0',
        description='Minimum observation distance for prototype update node (meters)'
    )

    valid_projection_threshold_arg = DeclareLaunchArgument(
        'valid_projection_threshold',
        default_value='0.95',
        description='Valid projection threshold for footprint visibility [0.0-1.0]'
    )

    # Get launch configurations
    class_name = LaunchConfiguration('class_name')
    threshold = LaunchConfiguration('threshold')
    device = LaunchConfiguration('device')
    input_size = LaunchConfiguration('input_size')
    prototype_dir = LaunchConfiguration('prototype_dir')
    momentum = LaunchConfiguration('momentum')
    update_interval = LaunchConfiguration('update_interval')
    image_topic = LaunchConfiguration('image_topic')
    debug = LaunchConfiguration('debug')
    min_observation_distance = LaunchConfiguration('min_observation_distance')
    valid_projection_threshold = LaunchConfiguration('valid_projection_threshold')

    # Segmentation node
    segmentation_node = Node(
        package='tarts_ros',
        executable='tarts_segmentation',
        name='tarts_segmentation',
        output='screen',
        parameters=[{
            'class_name': class_name,
            'threshold': threshold,
            'device': device,
            'input_size': input_size,
            'prototype_dir': prototype_dir,
            'slic_n_segments': 400,
            'slic_compactness': 30.0,
            'image_topic': image_topic,
            'visualization_alpha': 0.5,
            'debug': debug,
        }],
        remappings=[
            ('/camera/color/image_raw', image_topic),
        ]
    )

    # Prototype update node
    prototype_update_node = Node(
        package='tarts_ros',
        executable='tarts_prototype_update',
        name='tarts_prototype_update',
        output='screen',
        parameters=[{
            'class_name': class_name,
            'device': device,
            'prototype_dir': prototype_dir,
            'momentum': momentum,
            'update_interval': update_interval,
            'cache_interval': 0.1,
            'cache_max_count': 30,
            'robot_length': 0.627,
            'robot_width': 0.549,
            'robot_height': 0.248,
            'camera_translation': [0.2, 0.0, 0.41],
            'camera_rotation': [0.0, 0.0, 0.0, 1.0],  # quaternion (x, y, z, w)
            'debug': debug,
            # Prototype Update Node selection parameters
            'min_observation_distance': min_observation_distance,
            'valid_projection_threshold': valid_projection_threshold,
        }]
    )

    return LaunchDescription([
        # Declare arguments
        class_name_arg,
        threshold_arg,
        device_arg,
        input_size_arg,
        prototype_dir_arg,
        momentum_arg,
        update_interval_arg,
        image_topic_arg,
        debug_arg,
        min_observation_distance_arg,
        valid_projection_threshold_arg,

        # Launch nodes
        segmentation_node,
        prototype_update_node,
    ])
