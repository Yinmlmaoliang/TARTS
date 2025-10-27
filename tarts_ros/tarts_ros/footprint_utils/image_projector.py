from os.path import join
import torch
from torchvision import transforms as T
from kornia.geometry.camera.pinhole import PinholeCamera
from kornia.geometry.linalg import transform_points
from kornia.utils.draw import draw_convex_polygon
from liegroups.torch import SE3, SO3


class ImageProjector:
    def __init__(self, K: torch.tensor, h: int, w: int, new_h: int = None, new_w: int = None):
        """Initializes the projector using the pinhole model, without distortion

        Args:
            K (torch.Tensor, dtype=torch.float32, shape=(B, 4, 4)): Camera matrices
            pose_camera_in_world (torch.Tensor, dtype=torch.float32, shape=(B, 4, 4)): Extrinsics SE(3) matrix
            h (torch.Tensor, dtype=torch.int64): Image height
            w (torch.Tensor, dtype=torch.int64): Image width
            new_h (int): New height size
            new_w (int): New width size

        Returns:
            None
        """

        # TODO: Add shape checks

        # Get device for later
        device = K.device
        self._device = device

        # Initialize pinhole model (no extrinsics)
        E = torch.eye(4).expand(K.shape).to(device)

        # Store original parameters
        self.K = K
        self.height = h
        self.width = w

        new_h = self.height.item() if new_h is None else new_h

        # Compute scale
        sy = new_h / h
        sx = (new_w / w) if (new_w is not None) else sy

        # Compute scaled parameters
        sh = new_h
        sw = new_w if new_w is not None else sh

        # Prepare image cropper
        if new_w is None or new_w == new_h:
            self.image_crop = T.Compose([T.Resize(new_h, T.InterpolationMode.NEAREST), T.CenterCrop(new_h)])
        else:
            self.image_crop = T.Resize([new_h, new_w], T.InterpolationMode.NEAREST)

        # Adjust camera matrix
        # Fill values
        sK = K.clone()
        if new_w is None or new_w == new_h:
            sK[:, 0, 0] = K[:, 1, 1] * sy
            sK[:, 0, 2] = K[:, 1, 2] * sy
            sK[:, 1, 1] = K[:, 1, 1] * sy
            sK[:, 1, 2] = K[:, 1, 2] * sy
        else:
            sK[:, 0, 0] = K[:, 0, 0] * sx
            sK[:, 0, 2] = K[:, 0, 2] * sx
            sK[:, 1, 1] = K[:, 1, 1] * sy
            sK[:, 1, 2] = K[:, 1, 2] * sy

        # Initialize camera with scaled parameters
        sh = torch.IntTensor([sh]).to(device)
        sw = torch.IntTensor([sw]).to(device)
        self.camera = PinholeCamera(sK, E, sh, sw)

        # Preallocate masks
        B = self.camera.batch_size
        C = 3  # RGB channel output
        H = self.camera.height.item()
        W = self.camera.width.item()
        # Create output mask
        self.masks = torch.zeros((B, C, H, W), dtype=torch.float32, device=device)

    @property
    def scaled_camera_matrix(self):
        return self.camera.intrinsics.clone()[:3, :3]

    def change_device(self, device):
        """Changes the device of all the class members

        Args:
            device (str): new device
        """
        self.K = self.K.to(device)
        self.camera = PinholeCamera(
            self.camera.intrinsics.to(device),
            self.camera.extrinsics.to(device),
            self.camera.height.to(device),
            self.camera.width.to(device),
        )

    def check_validity(self, points_3d: torch.tensor, points_2d: torch.tensor) -> torch.tensor:
        """Check that the points are valid after projecting them on the image

        Args:
            points_3d: (torch.Tensor, dtype=torch.float32, shape=(B, N, 3)): B batches of N points in camera frame
            points_2d: (torch.Tensor, dtype=torch.float32, shape=(B, N, 2)): B batches of N points on the image

        Returns:
            valid_points: (torch.Tensor, dtype=torch.bool, shape=(B, N, 1)): B batches of N bools
        """

        # Check cheirality (if points are behind the camera, i.e, negative z)
        valid_z = points_3d[..., 2] >= 0
        # # Check if projection is within image range
        valid_xmin = points_2d[..., 0] >= 0
        valid_xmax = points_2d[..., 0] <= self.camera.width
        valid_ymin = points_2d[..., 1] >= 0
        valid_ymax = points_2d[..., 1] <= self.camera.height

        # Return validity
        return valid_z & valid_xmax & valid_xmin & valid_ymax & valid_ymin, valid_z

    def project(self, pose_camera_in_world: torch.tensor, points_W: torch.tensor):
        """Applies the pinhole projection model to a batch of points

        Args:
            points: (torch.Tensor, dtype=torch.float32, shape=(B, N, 3)): B batches of N input points in world frame

        Returns:
            projected_points: (torch.Tensor, dtype=torch.float32, shape=(B, N, 2)): B batches of N output points on image space
        """

        # Adjust input points depending on the extrinsics
        T_CW = pose_camera_in_world.inverse()
        # Convert from fixed to camera frame
        points_cam_ros = transform_points(T_CW, points_W)
        # (2) 定义 ROS->Kornia 相机坐标转换矩阵（旋转矩阵），将 ROS 坐标转换为光轴为 z 的相机坐标
        R_adjust = torch.tensor([[0, -1, 0],
                                [0,  0, -1],
                                [1,  0,  0]], dtype=torch.float32)
        H = torch.eye(4, dtype=torch.float32)
        H[:3, :3] = R_adjust
        H = H.unsqueeze(0).to(self._device)
    
        # (3) 利用 H 将 ROS 相机坐标转换为 Kornia 相机坐标（注意：变换顺序为先 T_wc 后 H）
        points_cam = transform_points(H, points_cam_ros)
        # print("相机坐标系：", points_cam)

        # Project points to image
        projected_points = self.camera.project(points_cam)
        # print("projected_points: ", projected_points)

        # Validity check (if points are out of the field of view)
        valid_points, valid_z = self.check_validity(points_cam, projected_points)

        # Return projected points and validity
        return projected_points, valid_points, valid_z

    def project_and_render(
        self,
        pose_camera_in_world: torch.tensor,
        points: torch.tensor,
        colors: torch.tensor,
        image: torch.tensor = None,
    ):
        """Projects the points and returns an image with the projection

        Args:
            points: (torch.Tensor, dtype=torch.float32, shape=(B, N, 3)): B batches, of N input points in 3D space
            colors: (torch.Tensor, rtype=torch.float32, shape=(B, 3))

        Returns:
            out_img (torch.tensor, dtype=torch.int64): Image with projected points
        """

        # self.masks = self.masks * 0.0
        B = self.camera.batch_size
        C = 3  # RGB channel output
        H = self.camera.height.item()
        W = self.camera.width.item()
        self.masks = torch.zeros((B, C, H, W), dtype=torch.float32, device=self.camera.camera_matrix.device)
        image_overlay = image

        # Project points
        projected_points, valid_points, valid_z = self.project(pose_camera_in_world, points)
        projected_points[~valid_z, :] = torch.nan
        # projected_points[projected_points < 0.0]

        # Fill the mask
        self.masks = draw_convex_polygon(self.masks, projected_points, colors)

        # Draw on image (if applies)
        if image is not None:
            if len(image.shape) != 4:
                image = image[None]
            image_overlay = draw_convex_polygon(image, projected_points, colors)

        # Return torch masks
        self.masks[self.masks == 0.0] = torch.nan
        binary_mask = ~torch.isnan(self.masks).all(dim=1)  # Shape: (B, H, W)
        
        return self.masks, image_overlay, projected_points, valid_points, binary_mask

    def resize_image(self, image: torch.tensor):
        return self.image_crop(image)