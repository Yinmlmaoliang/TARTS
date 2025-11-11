#!/usr/bin/env python3
"""
TARTS Segmentation Node for ROS2

Multi-threaded pipeline architecture with performance optimizations:
1. Image preprocessing thread: Resize images to 480x480
2. Feature extraction thread: DINO features + SLIC segmentation with patch mapping
   - Pre-computed pixel-to-patch mappings
   - CUDA stream async execution (DINO + SLIC overlap)
3. Matching thread: Cosine similarity matching and mask generation
   - Fully vectorized operations (no Python loops)

Performance: ~20-30ms end-to-end latency on CUDA
Results are interpolated back to original image size for visualization.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
import numpy as np
import cv2
import threading
import queue
import os
import time
import torch

# Import TARTS core components
from tarts_core.tarts.engine import SegmentationEngine
from tarts_core.prototype.manager import PrototypeManager
from tarts_core.utils.visualization import create_mask_overlay

# Import ROS image utilities
from .image_utils import ros_image_to_numpy, numpy_to_ros_image, mask_to_ros_image

# Import custom messages
from tarts_msgs.msg import FeatureData


class TartsSegmentationNode(Node):
    """TARTS segmentation pipeline with multi-threading."""

    def __init__(self):
        super().__init__('tarts_segmentation')

        # Declare parameters
        self.declare_parameter('class_name', 'corn')
        self.declare_parameter('threshold', 0.5)
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('input_size', 480)
        self.declare_parameter('prototype_dir', os.path.expanduser('~/.ros/tarts_prototypes'))
        self.declare_parameter('slic_n_segments', 400)
        self.declare_parameter('slic_compactness', 30.0)
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('visualization_alpha', 0.5)
        self.declare_parameter('debug', False)

        # Otsu threshold parameters
        self.declare_parameter('threshold_method', 'fixed')
        self.declare_parameter('otsu_nbins', 256)
        self.declare_parameter('otsu_method', 'standard')
        self.declare_parameter('otsu_sigma', 1.0)

        # Model parameters
        self.declare_parameter('backbone_type', 'vits16')
        self.declare_parameter('dropout_p', 0.0)

        # Get parameters
        self.class_name = self.get_parameter('class_name').value
        self.threshold = self.get_parameter('threshold').value
        device = self.get_parameter('device').value
        self.input_size = self.get_parameter('input_size').value
        prototype_dir = self.get_parameter('prototype_dir').value
        self.prototype_dir = prototype_dir  # Store for prototype reload
        self.slic_n_segments = self.get_parameter('slic_n_segments').value
        self.slic_compactness = self.get_parameter('slic_compactness').value
        image_topic = self.get_parameter('image_topic').value
        self.vis_alpha = self.get_parameter('visualization_alpha').value
        self.debug = self.get_parameter('debug').value

        # Get Otsu parameters
        self.threshold_method = self.get_parameter('threshold_method').value
        self.otsu_nbins = self.get_parameter('otsu_nbins').value
        self.otsu_method = self.get_parameter('otsu_method').value
        self.otsu_sigma = self.get_parameter('otsu_sigma').value

        # Get model parameters
        self.backbone_type = self.get_parameter('backbone_type').value
        self.dropout_p = self.get_parameter('dropout_p').value

        # Check device availability
        if device == 'cuda' and not torch.cuda.is_available():
            self.get_logger().warn('CUDA not available, falling back to CPU')
            device = 'cpu'
        self.device = device

        self.get_logger().info('=' * 60)
        self.get_logger().info('TARTS Segmentation Node Initialization')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'Class name: {self.class_name}')
        self.get_logger().info(f'Model backbone: {self.backbone_type}')
        self.get_logger().info(f'Dropout probability: {self.dropout_p}')
        self.get_logger().info(f'Threshold method: {self.threshold_method}')
        if self.threshold_method == 'fixed':
            self.get_logger().info(f'Fixed threshold: {self.threshold}')
        else:
            self.get_logger().info(f'Otsu method: {self.otsu_method}, nbins: {self.otsu_nbins}')
        self.get_logger().info(f'Device: {self.device}')
        self.get_logger().info(f'Input size: {self.input_size}')
        self.get_logger().info(f'SLIC segments: {self.slic_n_segments}')

        # Initialize TARTS segmentation engine
        self.get_logger().info('Initializing TARTS segmentation engine...')
        self.engine = SegmentationEngine(
            input_size=self.input_size,
            device=self.device,
            backbone_type=self.backbone_type,
            slic_n_segments=self.slic_n_segments,
            slic_compactness=self.slic_compactness,
            dropout_p=self.dropout_p,
            threshold_method=self.threshold_method,
            otsu_nbins=self.otsu_nbins,
            otsu_method=self.otsu_method,
            otsu_sigma=self.otsu_sigma
        )

        engine_info = self.engine.get_info()
        self.get_logger().info(f"Engine info: {engine_info['backbone']}, "
                              f"feature_dim={engine_info['feature_dim']}, "
                              f"patch_size={engine_info['patch_size']}")

        # Load prototype
        prototype_path = os.path.join(prototype_dir, f'{self.class_name}.pt')
        self.get_logger().info(f'Loading prototype: {prototype_path}')
        self.prototype_manager = PrototypeManager(device=self.device)
        self.prototype = self.prototype_manager.load(prototype_path, verbose=False)
        self.get_logger().info(f'Prototype loaded successfully')

        # Threading queues
        self.preprocess_queue = queue.Queue(maxsize=1)
        self.feature_queue = queue.Queue(maxsize=10)
        self.running = True

        # Statistics
        self.frame_count = 0
        self.lock = threading.Lock()

        # Start worker threads
        self.feature_thread = threading.Thread(target=self._feature_extraction_worker, daemon=True)
        self.matching_thread = threading.Thread(target=self._matching_worker, daemon=True)
        self.feature_thread.start()
        self.matching_thread.start()

        # ROS subscribers and publishers
        self.image_sub = self.create_subscription(
            Image, image_topic, self._image_callback, 1)
        self.mask_pub = self.create_publisher(Image, '/tarts/mask', 10)
        self.vis_pub = self.create_publisher(Image, '/tarts/visualized', 10)
        self.similarity_pub = self.create_publisher(Float32, '/tarts/similarity', 10)
        self.feature_pub = self.create_publisher(FeatureData, '/tarts/features', 10)

        # Subscribe to prototype updates (for online adaptation)
        self.prototype_update_sub = self.create_subscription(
            Float32,  # Placeholder - will receive update notifications
            '/tarts/prototype_updated',
            self._prototype_update_callback,
            10
        )

        self.get_logger().info('=' * 60)
        self.get_logger().info('Pipeline ready!')
        self.get_logger().info('=' * 60)

    def _image_callback(self, msg):
        """ROS image callback - preprocessing thread (main thread).

        Only performs resize operation. Image is kept as uint8 for direct SLIC usage.
        """
        try:
            t_start = time.time()

            # Convert ROS image to numpy (RGB, uint8)
            np_img = ros_image_to_numpy(msg, desired_encoding='rgb8')
            original_size = (np_img.shape[0], np_img.shape[1])  # (H, W)

            # Resize to input_size (using cv2 for speed, returns uint8)
            t_resize_start = time.time()
            np_img_resized = cv2.resize(np_img, (self.input_size, self.input_size),
                                        interpolation=cv2.INTER_LINEAR)
            t_resize = (time.time() - t_resize_start) * 1000  # ms

            if self.debug:
                self.get_logger().info(f'[Preprocessing] Resize: {t_resize:.2f}ms')

            # Try to put into queue (non-blocking)
            try:
                self.preprocess_queue.put_nowait((np_img_resized, original_size, msg.header))
                with self.lock:
                    if self.frame_count == 0:
                        self.get_logger().info(f'First frame received: {original_size}, queue size: {self.preprocess_queue.qsize()}')
            except queue.Full:
                # Silently drop frame - this is expected behavior to ensure we use latest image
                pass

            # Small delay to reduce CPU usage (preprocessing is much faster than feature extraction)
            time.sleep(0.001)  # 10ms delay

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')
            import traceback
            traceback.print_exc()

    def _feature_extraction_worker(self):
        """Feature extraction and SLIC segmentation worker thread."""
        self.get_logger().info('Feature extraction thread started')
        first_frame = True

        while self.running:
            try:
                # Get preprocessed image from queue (with timeout)
                np_img_resized, original_size, header = self.preprocess_queue.get(timeout=0.1)

                if first_frame:
                    self.get_logger().info(f'Feature thread processing first frame: {np_img_resized.shape}')
                    first_frame = False

                t_total_start = time.time()

                # Extract features and sparsify using TARTS engine
                sparse_features, seg = self.engine.extract_features_and_sparsify(np_img_resized)

                t_total = (time.time() - t_total_start) * 1000  # ms

                if self.debug:
                    self.get_logger().info(
                        f'[Feature Extraction & Superpixel Generation] {t_total:.2f}ms'
                    )

                # Put results into feature queue
                try:
                    self.feature_queue.put_nowait((sparse_features, seg, np_img_resized, original_size, header))
                except queue.Full:
                    self.get_logger().warn('Feature queue full, dropping frame')

                self.preprocess_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error in feature extraction: {e}')
                import traceback
                traceback.print_exc()

    def _matching_worker(self):
        """Matching and segmentation worker thread."""
        self.get_logger().info('Matching thread started')
        first_frame = True

        while self.running:
            try:
                # Get features from queue (with timeout)
                sparse_features, seg, np_img, original_size, header = self.feature_queue.get(timeout=0.1)

                if first_frame:
                    self.get_logger().info(f'Matching thread processing first frame, features: {sparse_features.shape}')
                    first_frame = False

                # Publish feature data BEFORE matching (for prototype update node)
                self._publish_features(sparse_features, seg, original_size, header)

                # Match and generate mask using TARTS engine
                t_match_start = time.time()
                # Use current prototype (may be updated by prototype_update_node)
                with self.lock:
                    current_prototype = self.prototype_manager.get_prototype()
                mask_original, avg_similarity = self.engine.match_and_generate_mask(
                    sparse_features, seg, current_prototype, self.threshold, original_size
                )
                t_match = (time.time() - t_match_start) * 1000  # ms

                # Create visualization
                t_vis_start = time.time()
                # Resize np_img to original size for visualization
                np_img_original = cv2.resize(np_img, (original_size[1], original_size[0]))
                vis_img = create_mask_overlay(
                    np_img_original,
                    mask_original,
                    color=(0, 255, 0),
                    alpha=self.vis_alpha
                )
                t_vis = (time.time() - t_vis_start) * 1000  # ms

                if self.debug:
                    self.get_logger().info(
                        f'[Aggregation & Comparison] {t_match:.2f}ms | '
                        f'Visualization: {t_vis:.2f}ms'
                    )

                # Publish results
                self._publish_results(mask_original, vis_img, avg_similarity, header)

                # Update statistics
                with self.lock:
                    self.frame_count += 1
                    if self.frame_count % 30 == 0:
                        self.get_logger().info(f'Processed {self.frame_count} frames, avg similarity: {avg_similarity:.4f}')

                self.feature_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                self.get_logger().error(f'Error in matching: {e}')
                import traceback
                traceback.print_exc()

    def _publish_features(self, sparse_features, seg_tensor, original_size, header):
        """Publish feature data for prototype update node."""
        try:
            feature_msg = FeatureData()
            feature_msg.header = header

            # Flatten and convert sparse features
            feature_msg.sparse_features = sparse_features.cpu().flatten().numpy().tolist()
            feature_msg.sparse_features_shape = [int(d) for d in sparse_features.shape]

            # Flatten and convert segmentation tensor
            feature_msg.seg_tensor = seg_tensor.cpu().flatten().int().numpy().tolist()
            feature_msg.seg_tensor_shape = [int(d) for d in seg_tensor.shape]

            # Add image metadata
            feature_msg.original_height = int(original_size[0])
            feature_msg.original_width = int(original_size[1])
            feature_msg.resized_size = int(self.input_size)

            self.feature_pub.publish(feature_msg)

        except Exception as e:
            self.get_logger().error(f'Error publishing features: {e}')

    def _prototype_update_callback(self, msg):
        """Handle prototype update notifications from prototype_update_node."""
        # The prototype is updated in PrototypeManager, just log notification
        with self.lock:
            self.get_logger().info('Prototype updated by online adaptation')
            # Reload the updated prototype
            prototype_path = os.path.join(self.prototype_dir, f'{self.class_name}.pt')
            self.get_logger().info(f'Reloading prototype from: {prototype_path}')
            self.prototype = self.prototype_manager.load(prototype_path, verbose=False)
            self.get_logger().info(f'Prototype reloaded successfully')

    def _publish_results(self, mask, vis_img, avg_similarity, header):
        """Publish segmentation results."""
        # Publish binary mask
        mask_msg = mask_to_ros_image(mask, encoding='mono8')
        mask_msg.header = header
        self.mask_pub.publish(mask_msg)

        # Publish visualization
        vis_msg = numpy_to_ros_image(vis_img, desired_encoding='rgb8')
        vis_msg.header = header
        self.vis_pub.publish(vis_msg)

        # Publish average similarity
        sim_msg = Float32()
        sim_msg.data = float(avg_similarity)
        self.similarity_pub.publish(sim_msg)

    def destroy_node(self):
        """Clean up when shutting down."""
        try:
            self.get_logger().info('Shutting down segmentation pipeline...')
        except Exception:
            print('Shutting down segmentation pipeline...')

        self.running = False

        # Wait for threads to finish
        if hasattr(self, 'feature_thread') and self.feature_thread.is_alive():
            self.feature_thread.join(timeout=1.0)
        if hasattr(self, 'matching_thread') and self.matching_thread.is_alive():
            self.matching_thread.join(timeout=1.0)

        try:
            super().destroy_node()
        except Exception:
            pass  # Context may already be invalid


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = None

    try:
        node = TartsSegmentationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        # Clean shutdown sequence
        if node is not None:
            try:
                node.destroy_node()
            except Exception as e:
                print(f'Error destroying node: {e}')

        # Only shutdown if context is still valid
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass  # Context already shutdown, ignore


if __name__ == '__main__':
    main()
