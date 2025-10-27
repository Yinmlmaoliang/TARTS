#!/usr/bin/env python3
"""
TARTS Prototype Update Node for ROS2

Online prototype adaptation using robot footprint projection.
Subscribes to features and odometry, projects robot footprint onto images,
and updates the prototype using positive samples from footprint regions.

Architecture:
1. Subscribe to /tarts/features and /odom (time-synchronized)
2. Maintain node cache for historical poses (0.1m interval, FIFO queue)
3. Every update_interval (0.5m), generate footprint from current and previous poses
4. Project footprint to image plane using camera intrinsics
5. Extract positive features from footprint regions
6. Update prototype using exponential moving average
7. Notify segmentation_node of update

Key Features:
- Time synchronization between features and odometry
- TF-based pose estimation (odom->base_link)
- Footprint projection with resize alignment (480x480)
- Momentum-based prototype adaptation
- Thread-safe prototype updates
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import CameraInfo
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32
import numpy as np
import torch
import threading
import os
import time
import math
from collections import deque

# Import message filters for synchronization
import message_filters

# Import TF2
import tf2_ros

# Import custom messages
from tarts_msgs.msg import FeatureData

# Import TARTS core components
from tarts_core.prototype.manager import PrototypeManager

# Import footprint projection utilities
from .footprint_utils import ImageProjector, DataNode


class PrototypeUpdateNode(Node):
    """
    Online prototype update node using footprint projection.
    """

    def __init__(self):
        super().__init__('tarts_prototype_update')

        # Declare parameters
        self.declare_parameter('class_name', 'corn')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('prototype_dir', os.path.expanduser('~/.ros/tarts_prototypes'))
        self.declare_parameter('momentum', 0.9)
        self.declare_parameter('update_interval', 0.5)  # meters
        self.declare_parameter('cache_interval', 0.1)   # meters
        self.declare_parameter('cache_max_count', 30)
        self.declare_parameter('robot_length', 0.627)
        self.declare_parameter('robot_width', 0.549)
        self.declare_parameter('robot_height', 0.248)
        self.declare_parameter('camera_translation', [0.2, 0.0, 0.41])
        self.declare_parameter('camera_rotation', [0.0, 0.0, 0.0, 1.0])  # quat (x,y,z,w)
        self.declare_parameter('debug', False)

        # Prototype Update Node selection parameters
        self.declare_parameter('min_observation_distance', 1.0)
        self.declare_parameter('valid_projection_threshold', 0.95)

        # Get parameters
        self.class_name = self.get_parameter('class_name').value
        device = self.get_parameter('device').value
        prototype_dir = self.get_parameter('prototype_dir').value
        self.prototype_dir = prototype_dir  # Store for saving updated prototype
        self.momentum = self.get_parameter('momentum').value
        self.update_interval = self.get_parameter('update_interval').value
        self.cache_interval = self.get_parameter('cache_interval').value
        self.cache_max_count = self.get_parameter('cache_max_count').value
        self.robot_length = self.get_parameter('robot_length').value
        self.robot_width = self.get_parameter('robot_width').value
        self.robot_height = self.get_parameter('robot_height').value
        self.camera_translation = self.get_parameter('camera_translation').value
        self.camera_rotation = self.get_parameter('camera_rotation').value
        self.debug = self.get_parameter('debug').value

        # Get prototype update node selection parameters
        self.min_observation_distance = self.get_parameter('min_observation_distance').value
        self.valid_projection_threshold = self.get_parameter('valid_projection_threshold').value

        # Check device availability
        if device == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn('CUDA not available, falling back to CPU')
            device = 'cpu'
        self.device = device

        self.get_logger().info('=' * 60)
        self.get_logger().info('TARTS Prototype Update Node Initialization')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Class name: {self.class_name}')
        self.get_logger().info(f'Momentum: {self.momentum}')
        self.get_logger().info(f'Update interval: {self.update_interval}m')
        self.get_logger().info(f'Device: {self.device}')

        # Initialize TF buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Camera intrinsics (will be received from topic)
        self.camera_intrinsics = None
        self.camera_info_received = False

        # Tracking state
        self.cumulative_distance = 0.0
        self.last_cached_distance = 0.0
        self.last_updated_distance = 0.0
        self.last_odom_position = None

        # Node cache for footprint generation
        self.data_nodes = deque(maxlen=self.cache_max_count)
        self.lock = threading.Lock()

        # Statistics
        self.update_count = 0
        self.total_positive_samples = 0

        # Load prototype
        prototype_path = os.path.join(prototype_dir, f'{self.class_name}.pt')
        self.get_logger().info(f'Loading initial prototype: {prototype_path}')
        self.prototype_manager = PrototypeManager(device=self.device)
        self.prototype_manager.load(prototype_path, verbose=False)
        self.get_logger().info('Initial prototype loaded successfully')

        # Create subscribers with message filters for synchronization
        self.feature_sub = message_filters.Subscriber(self, FeatureData, '/tarts/features')
        self.odom_sub = message_filters.Subscriber(self, Odometry, '/odom')

        # Time synchronizer (5ms tolerance)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.feature_sub, self.odom_sub],
            queue_size=10,
            slop=0.05  # 50ms tolerance
        )
        self.sync.registerCallback(self.synchronized_callback)

        # Subscribe to camera info (one-time)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            10
        )

        # Publisher for update notifications
        self.update_pub = self.create_publisher(Float32, '/tarts/prototype_updated', 10)

        self.get_logger().info('=' * 60)
        self.get_logger().info('Prototype update node ready!')
        self.get_logger().info('=' * 60)

    def camera_info_callback(self, msg):
        """Receive camera intrinsics (one-time)."""
        if not self.camera_info_received:
            self.camera_intrinsics = self._extract_camera_intrinsics(msg)
            if self.camera_intrinsics is not None:
                self.camera_info_received = True
                self.get_logger().info('Camera intrinsics received')
                # Unsubscribe after receiving
                self.destroy_subscription(self.camera_info_sub)

    def synchronized_callback(self, feature_msg, odom_msg):
        """
        Handle synchronized feature and odometry messages.

        This is the main processing loop:
        1. Update cumulative distance
        2. Cache nodes at regular intervals
        3. Update prototype when update_interval is reached
        """
        try:
            # Check if camera intrinsics are available
            if not self.camera_info_received:
                return

            # Extract timestamp
            timestamp = feature_msg.header.stamp.sec + feature_msg.header.stamp.nanosec / 1e9

            # Update cumulative distance
            self._update_cumulative_distance(odom_msg)

            # Cache node if interval reached
            if (self.cumulative_distance - self.last_cached_distance) >= self.cache_interval:
                self._cache_current_node(timestamp, feature_msg, odom_msg)

            # Update prototype if interval reached and we have enough nodes
            if (self.cumulative_distance - self.last_updated_distance) >= self.update_interval:
                if len(self.data_nodes) >= 2:  # Need at least 2 nodes for footprint
                    self._update_prototype(timestamp)

        except Exception as e:
            self.get_logger().error(f'Error in synchronized callback: {e}')
            import traceback
            traceback.print_exc()

    def _update_cumulative_distance(self, odom_msg):
        """Update cumulative distance traveled."""
        try:
            position = odom_msg.pose.pose.position
            current_position = np.array([position.x, position.y, position.z])

            if self.last_odom_position is not None:
                distance_increment = np.linalg.norm(current_position - self.last_odom_position)
                self.cumulative_distance += distance_increment

            self.last_odom_position = current_position

        except Exception as e:
            self.get_logger().warn(f'Error updating cumulative distance: {e}')

    def _cache_current_node(self, timestamp, feature_msg, odom_msg):
        """
        Cache current node with pose, features, and image projector.
        """
        try:
            # Get robot pose from TF
            pose_base_in_world = self._get_pose_from_tf(timestamp)
            if pose_base_in_world is None:
                return

            # Create DataNode
            data_node = self._create_data_node(
                timestamp=timestamp,
                pose_base_in_world=pose_base_in_world,
                feature_msg=feature_msg
            )

            if data_node is not None:
                # Add distance metadata
                data_node.cumulative_distance = self.cumulative_distance

                # Add to cache (FIFO)
                with self.lock:
                    self.data_nodes.append(data_node)

                self.last_cached_distance = self.cumulative_distance

        except Exception as e:
            self.get_logger().error(f'Error caching node: {e}')

    def _update_prototype(self, current_timestamp):
        """
        Update prototype using footprint projection.

        Steps:
        1. Get current and previous nodes
        2. Generate footprint polygon
        3. Project footprint to image plane
        4. Extract positive features from footprint regions
        5. Update prototype with momentum
        """
        try:
            t_start = time.time()

            # Get current and previous nodes
            with self.lock:
                if len(self.data_nodes) < 2:
                    return

                current_node = self.data_nodes[-1]
                # Find node approximately update_interval meters ago
                target_distance = self.cumulative_distance - self.update_interval
                prev_node = self._find_closest_node(target_distance)

            if prev_node is None:
                return

            # Generate footprint
            footprint, points, dimensions_info = current_node.make_footprint_with_node(prev_node)

            # Select optimal prototype update node for projection
            optimal_prototype_update_node = self._select_optimal_prototype_update_from_cache(
                current_node, prev_node, footprint
            )

            if optimal_prototype_update_node is None:
                return

            # Project footprint to image plane using optimal prototype update node
            pose_batch = optimal_prototype_update_node.pose_cam_in_world.unsqueeze(0)
            color = torch.ones((3,), device=footprint.device)

            mask, _, _, _, bi_mask = optimal_prototype_update_node.image_projector.project_and_render(
                pose_batch, footprint[None], color
            )

            # Check if footprint is visible
            if bi_mask is None or bi_mask.sum() == 0:
                return

            # Extract positive features from footprint regions
            # Note: Must use optimal_prototype_update_node's features since bi_mask is in its coordinate frame
            pos_features = self._extract_positive_features(
                bi_mask, optimal_prototype_update_node.seg_tensor, optimal_prototype_update_node.sparse_features
            )

            if pos_features.shape[0] == 0:
                return

            # Update prototype with momentum
            with self.lock:
                self.prototype_manager.update_with_momentum(pos_features, momentum=self.momentum)
                self.update_count += 1
                self.total_positive_samples += pos_features.shape[0]

            self.last_updated_distance = self.cumulative_distance

            # Notify segmentation node
            update_msg = Float32()
            update_msg.data = float(self.update_count)
            self.update_pub.publish(update_msg)

            # Save prototype after each update
            save_path = os.path.join(self.prototype_dir, f'{self.class_name}.pt')
            self.prototype_manager.save(
                prototype=self.prototype_manager.get_prototype(),
                save_path=save_path,
                verbose=False  # Disable print output, use ROS logger instead
            )

            t_elapsed = (time.time() - t_start) * 1000

            self.get_logger().info(
                f'Prototype updated and saved #{self.update_count} - '
                f'Positive samples: {pos_features.shape[0]}, '
                f'Total: {self.total_positive_samples}, '
                f'Time: {t_elapsed:.1f}ms'
            )

        except Exception as e:
            self.get_logger().error(f'Error updating prototype: {e}')
            import traceback
            traceback.print_exc()

    def _extract_positive_features(self, bi_mask, seg_tensor, sparse_features):
        """
        Extract positive features from footprint mask.

        Args:
            bi_mask: (H, W) or (B, H, W) binary footprint mask
            seg_tensor: (H, W) segment labels
            sparse_features: (N_segments, C) sparse features

        Returns:
            pos_features: (M, C) positive sample features
        """
        try:
            # Remove batch dimension if present
            if bi_mask.dim() == 3:
                bi_mask = bi_mask.squeeze(0)  # (B, H, W) -> (H, W)

            # Ensure on same device
            bi_mask = bi_mask.to(seg_tensor.device)

            # Get segment IDs covered by footprint
            footprint_segments = seg_tensor[bi_mask > 0.5]

            if footprint_segments.numel() == 0:
                return torch.empty((0, sparse_features.shape[1]), device=sparse_features.device)

            # Get unique segment IDs
            unique_segments = torch.unique(footprint_segments)

            # Extract corresponding features
            pos_features = sparse_features[unique_segments]

            return pos_features

        except Exception as e:
            self.get_logger().error(f'Error extracting positive features: {e}')
            self.get_logger().error(
                f'Tensor shapes - bi_mask: {bi_mask.shape}, '
                f'seg_tensor: {seg_tensor.shape}, '
                f'sparse_features: {sparse_features.shape}'
            )
            import traceback
            traceback.print_exc()
            return torch.empty((0, sparse_features.shape[1]), device=sparse_features.device)

    def _select_optimal_prototype_update_from_cache(self, current_node, prev_node, footprint):
        """
        Select optimal prototype update node from cache based on footprint projection validity.

        This method finds the first suitable node for projecting the footprint by:
        1. Ensuring time causality (timestamp < prev_node.timestamp)
        2. Ensuring minimum observation distance (≥1m from footprint center)
        3. Finding first node with valid projection ratio ≥ 95%

        Args:
            current_node: Current extract node
            prev_node: Previous node for footprint generation
            footprint: The footprint polygon (N, 2)

        Returns:
            DataNode: First suitable prototype update node or None if no suitable node found
        """
        try:
            # Calculate footprint center position
            current_pos = current_node.pose_base_in_world[:3, 3]
            prev_pos = prev_node.pose_base_in_world[:3, 3]
            footprint_center = (current_pos + prev_pos) / 2

            # Build candidate list: filter by time causality and minimum observation distance
            candidates = []
            for node in self.data_nodes:
                if node.timestamp < prev_node.timestamp:  # Time causality check
                    node_pos = node.pose_base_in_world[:3, 3]
                    distance_to_footprint = torch.norm(node_pos - footprint_center).item()

                    # Only consider nodes meeting minimum observation distance
                    if distance_to_footprint >= self.min_observation_distance:
                        candidates.append((node, distance_to_footprint))

            # Sort candidates by distance (search from minimum observation distance)
            candidates.sort(key=lambda x: x[1])

            # Search for first node with valid projection ratio ≥ threshold
            for node, distance in candidates:
                try:
                    # Test footprint projection validity on this node
                    pose_batch = node.pose_cam_in_world.unsqueeze(0)
                    projected_points, valid_points, valid_z = node.image_projector.project(
                        pose_batch, footprint[None]
                    )

                    # Calculate valid projection point ratio
                    total_points = footprint.shape[0]
                    valid_count = valid_points.sum().item() if valid_points is not None else 0
                    valid_ratio = valid_count / total_points if total_points > 0 else 0.0

                    # Return first node meeting the threshold
                    if valid_ratio >= self.valid_projection_threshold:
                        return node

                except Exception:
                    continue

            # No suitable node found
            return None

        except Exception as e:
            self.get_logger().error(f"Error selecting optimal prototype update from cache: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _find_closest_node(self, target_distance):
        """Find node closest to target distance."""
        if not self.data_nodes:
            return None

        closest_node = None
        min_distance_diff = float('inf')

        for node in self.data_nodes:
            if hasattr(node, 'cumulative_distance'):
                distance_diff = abs(node.cumulative_distance - target_distance)
                if distance_diff < min_distance_diff:
                    min_distance_diff = distance_diff
                    closest_node = node

        return closest_node

    def _get_pose_from_tf(self, timestamp):
        """
        Get odom->base_link transform from TF.

        Returns:
            torch.Tensor: 4x4 SE(3) transformation matrix or None
        """
        try:
            # Convert timestamp to ROS time
            from rclpy.time import Time
            if isinstance(timestamp, float):
                timestamp_ns = int(timestamp * 1e9)
                ros_time = Time(nanoseconds=timestamp_ns)
            else:
                ros_time = timestamp

            # Lookup transform
            transform = self.tf_buffer.lookup_transform(
                'odom', 'base_link', ros_time,
                timeout=Duration(seconds=0.1)
            )

            # Extract translation and rotation
            t = transform.transform.translation
            r = transform.transform.rotation

            # Build SE(3) matrix
            pose = torch.eye(4, dtype=torch.float32)
            pose[0, 3] = t.x
            pose[1, 3] = t.y
            pose[2, 3] = t.z

            # Quaternion to rotation matrix
            qx, qy, qz, qw = r.x, r.y, r.z, r.w
            R = torch.tensor([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
            ], dtype=torch.float32)

            pose[:3, :3] = R
            return pose

        except Exception as e:
            self.get_logger().debug(f'TF lookup failed: {e}')
            return None

    def _create_pose_cam_in_base(self):
        """Create camera extrinsics in base_link frame."""
        pose = torch.eye(4, dtype=torch.float32)

        # Set translation
        pose[0, 3] = self.camera_translation[0]
        pose[1, 3] = self.camera_translation[1]
        pose[2, 3] = self.camera_translation[2]

        # Set rotation from quaternion
        qx, qy, qz, qw = self.camera_rotation
        R = torch.tensor([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
        ], dtype=torch.float32)

        pose[:3, :3] = R
        return pose

    def _extract_camera_intrinsics(self, camera_info_msg):
        """Extract camera intrinsics from CameraInfo message."""
        try:
            K_list = camera_info_msg.k
            K = np.array(K_list).reshape(3, 3)

            # Convert to 4x4 homogeneous
            K_homogeneous = np.eye(4)
            K_homogeneous[:3, :3] = K

            return torch.tensor(K_homogeneous, dtype=torch.float32)

        except Exception as e:
            self.get_logger().error(f'Error extracting camera intrinsics: {e}')
            return None

    def _create_data_node(self, timestamp, pose_base_in_world, feature_msg):
        """
        Create DataNode for caching.

        Args:
            timestamp: float, timestamp in seconds
            pose_base_in_world: torch.Tensor (4, 4)
            feature_msg: FeatureData message

        Returns:
            DataNode or None
        """
        try:
            # Parse feature data
            sparse_features = torch.tensor(
                feature_msg.sparse_features,
                dtype=torch.float32
            ).reshape(tuple(feature_msg.sparse_features_shape))

            seg_tensor = torch.tensor(
                feature_msg.seg_tensor,
                dtype=torch.int64
            ).reshape(tuple(feature_msg.seg_tensor_shape))

            # Get image dimensions
            image_height = feature_msg.resized_size  # Use resized dimensions for alignment
            image_width = feature_msg.resized_size

            # Create camera extrinsics
            pose_cam_in_base = self._create_pose_cam_in_base()

            # Create ImageProjector (using resized dimensions for feature alignment)
            K_batch = self.camera_intrinsics.unsqueeze(0)
            h_tensor = torch.tensor(image_height, dtype=torch.int64)
            w_tensor = torch.tensor(image_width, dtype=torch.int64)
            image_projector = ImageProjector(
                K_batch, h_tensor, w_tensor,
                new_h=image_height, new_w=image_width
            )

            # Create DataNode
            node = DataNode(
                timestamp=timestamp,
                pose_base_in_world=pose_base_in_world,
                pose_cam_in_base=pose_cam_in_base,
                image=None,  # Don't need image data
                image_projector=image_projector,
                length=self.robot_length,
                width=self.robot_width,
                height=self.robot_height
            )

            # Attach feature data as attributes
            node.sparse_features = sparse_features
            node.seg_tensor = seg_tensor

            return node

        except Exception as e:
            self.get_logger().error(f'Error creating DataNode: {e}')
            import traceback
            traceback.print_exc()
            return None


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = None

    try:
        node = PrototypeUpdateNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except Exception:
                pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
