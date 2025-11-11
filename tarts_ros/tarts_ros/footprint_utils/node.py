from .image_projector import ImageProjector
from .utils import (
    make_box,
    make_plane,
    make_polygon_from_points,
    make_dense_plane,
)
from liegroups.torch import SE3, SO3

import os
import torch
from typing import Optional


class BaseNode:
    """Base node data structure"""

    _name = "base_node"

    def __init__(self, timestamp: float = 0.0, pose_base_in_world: torch.tensor = torch.eye(4)):
        assert isinstance(pose_base_in_world, torch.Tensor)

        self._timestamp = timestamp
        self._pose_base_in_world = pose_base_in_world
    def __str__(self):
        return f"{self._name}_{self._timestamp}"
    def __hash__(self):
        return hash(str(self))
    def __eq__(self, other):
        if other is None:
            return False
        return (
            self._name == other.name
            and self._timestamp == other.timestamp
            and torch.equal(self._pose_base_in_world, other.pose_base_in_world)
        )
    def __lt__(self, other):
        return self._timestamp < other.timestamp
    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self._pose_base_in_world = self._pose_base_in_world.to(device)
    @classmethod
    def from_node(cls, instance):
        return cls(timestamp=instance.timestamp, pose_base_in_world=instance.pose_base_in_world)
    def is_valid(self):
        return True
    def pose_between(self, other):
        """Computes pose difference (SE(3)) between this state and other
        Args:
            other (BaseNode): Other state
        Returns:
            tensor (torch.tensor): Pose difference expressed in this' frame
        """
        return other.pose_base_in_world.inverse() @ self.pose_base_in_world
    def distance_to(self, other):
        """Computes the relative distance between states
        Args:
            other (BaseNode): Other state
        Returns:
            distance (float): absolute distance between the states
        """
        # Compute pose difference, then log() to get a vector, then extract position coordinates, finally get norm
        return (
            SE3.from_matrix(
                self.pose_base_in_world.inverse() @ other.pose_base_in_world,
                normalize=True,
            )
            .log()[:3]
            .norm()
        )

    @property
    def name(self):
        return self._name

    @property
    def pose_base_in_world(self):
        return self._pose_base_in_world

    @property
    def timestamp(self):
        return self._timestamp

    @pose_base_in_world.setter
    def pose_base_in_world(self, pose_base_in_world: torch.tensor):
        self._pose_base_in_world = pose_base_in_world

    @timestamp.setter
    def timestamp(self, timestamp: float):
        self._timestamp = timestamp


class DataNode(BaseNode):
    """Mission node stores the minimum information required for traversability estimation
    All the information is stored on the image plane"""

    _name = "mission_node"

    def __init__(
        self,
        timestamp: float = 0.0,
        pose_base_in_world: torch.tensor = torch.eye(4),
        pose_cam_in_base: torch.tensor = torch.eye(4),
        pose_cam_in_world: torch.tensor = None,
        pose_footprint_in_base: torch.tensor = torch.eye(4),
        pose_footprint_in_world: torch.tensor = None,
        image: torch.tensor = None,
        image_projector: ImageProjector = None,
        length: float = 0.1,
        width: float = 0.1,
        height: float = 0.1,
    ):
        super().__init__(timestamp=timestamp, pose_base_in_world=pose_base_in_world)
        self._pose_base_in_world = pose_base_in_world
        self._pose_cam_in_base = pose_cam_in_base
        self._pose_footprint_in_base = pose_footprint_in_base
        self._pose_footprint_in_world = (
            pose_footprint_in_world if pose_footprint_in_world is not None
            else (
                self._pose_base_in_world @ self._pose_footprint_in_base
                if self._pose_footprint_in_base is not None
                else self._pose_base_in_world
            )
        )
        self._pose_cam_in_world = (
            self._pose_base_in_world @ self._pose_cam_in_base if pose_cam_in_world is None else pose_cam_in_world
        )
        self._image = image
        self._image_projector = image_projector
        self._length = length
        self._width = width
        self._height = height

        self._features = None
        self._prototype_update_mask = None
        self._patch_labels = None
        # print("pose base in world: ", self._pose_base_in_world)
        # print("pose camera in world: ", self._pose_cam_in_world)
        # print("pose footprint in world: ", self._pose_footprint_in_world)


    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        super().change_device(device)
        self._image_projector.change_device(device)

        self._pose_cam_in_base = self._pose_cam_in_base.to(device)
        self._pose_cam_in_world = self._pose_cam_in_world.to(device)

        if self._image is not None:
            self._image = self._image.to(device)
        if self._features is not None:
            self._features = self._features.to(device)

    def is_valid(self):
        valid = isinstance(self._features, torch.Tensor)
        return valid
    
    # def get_side_points(self):
    #     return make_plane(x=0.0, y=self._width, pose=self._pose_footprint_in_world, grid_size=2).to(
    #         self._pose_footprint_in_world.device
    #     )
    def get_side_points(self):
        # 计算机器人前侧的位姿：从基础位姿沿机器人前向（x轴正方向）平移半个机器人长度
        front_offset = torch.tensor([self._length / 2, 0, 0], device=self._pose_base_in_world.device)
        front_pose = self._pose_base_in_world.clone()
        front_pose[:3, 3] = front_pose[:3, 3] + front_pose[:3, :3] @ front_offset
        # 生成以机器人前侧为参考的边界点，注意x=0.0表示仅在机器人前侧生成一条直线，其两端为左右边界
        return make_plane(x=0.0, y=self._width, pose=front_pose, grid_size=2).to(self._pose_base_in_world.device)
    
    def make_footprint_with_node(self, other: BaseNode, grid_size: int = 10):
        # Get side points
        other_side_points = other.get_side_points()
        this_side_points = self.get_side_points()
        # swap points to make them counterclockwise
        this_side_points[[0, 1]] = this_side_points[[1, 0]]
        # The idea is to make a polygon like:
        # tsp[1] ---- tsp[0]
        #  |            |
        # osp[0] ---- osp[1]
        # with 'tsp': this_side_points and 'osp': other_side_points
        # 计算足迹的长度和宽度
        # 计算this_side_points的中点和other_side_points的中点
        this_midpoint = torch.mean(this_side_points, dim=0)
        other_midpoint = torch.mean(other_side_points, dim=0)
        
        # 计算长度（两个中点之间的距离）
        length_norm = torch.norm(this_midpoint - other_midpoint)
        footprint_length = length_norm.item() if hasattr(length_norm, 'item') else float(length_norm)
        
        # 计算宽度（this_side_points两点之间的距离）
        width_norm = torch.norm(this_side_points[0] - this_side_points[1])
        footprint_width = width_norm.item() if hasattr(width_norm, 'item') else float(width_norm)
        
        # 创建足迹尺寸信息字典
        dimensions_info = {
            "length": footprint_length,
            "width": footprint_width,

            }
        # Concat points to define the polygon
        points = torch.concat((this_side_points, other_side_points), dim=0)
        # 打印数据形状类型
        # print("points:", points.shape, points.dtype, points)
        # Make footprint
        footprint = make_polygon_from_points(points, grid_size=grid_size)
        return footprint, points, dimensions_info

    @property
    def camera_name(self):
        return self._camera_name

    @property
    def features(self):
        return self._features

    @property
    def image(self):
        return self._image

    @property
    def image_projector(self):
        return self._image_projector

    @property
    def pose_cam_in_world(self):
        return self._pose_cam_in_world
    
    @property
    def prototype_update_mask(self):
        return self._prototype_update_mask
    
    @property
    def patch_labels(self):
        return self._patch_labels

    @camera_name.setter
    def camera_name(self, camera_name):
        self._camera_name = camera_name

    @features.setter
    def features(self, features):
        self._features = features

    @image.setter
    def image(self, image):
        self._image = image

    @image_projector.setter
    def image_projector(self, image_projector):
        self._image_projector = image_projector

    @pose_cam_in_world.setter
    def pose_cam_in_world(self, pose_cam_in_world):
        self._pose_cam_in_world = pose_cam_in_world
    
    @prototype_update_mask.setter
    def prototype_update_mask(self, prototype_update_mask):
        self._prototype_update_mask = prototype_update_mask

    @patch_labels.setter
    def patch_labels(self, patch_labels):
        self._patch_labels = patch_labels
    

    def project_footprint(
        self,
        footprint: torch.tensor,
        color: torch.tensor = torch.FloatTensor([1.0, 1.0, 1.0]),
    ):
        (
            mask,
            image_overlay,
            projected_points,
            valid_points,
        ) = self._image_projector.project_and_render(self._pose_cam_in_world[None], footprint, color)

        return mask, image_overlay, projected_points, valid_points