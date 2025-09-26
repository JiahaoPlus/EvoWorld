# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
import numpy as np
import onnxruntime
import open3d as o3d
import requests
import torch
import trimesh
from scipy.spatial.transform import Rotation

import logging

logging.basicConfig(
    level=logging.INFO,  # or DEBUG
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# Constants
CUBEMAP_TRANSFORMS = {
    "front": np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]),
    "right": np.array([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]),
    "back": np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]),
    "left": np.array([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]),
    "top": np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]),
    "bottom": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
}

# Backward compatibility - keep old name
CUBEMAP = CUBEMAP_TRANSFORMS

# ONNX session configuration
options = onnxruntime.SessionOptions()
options.intra_op_num_threads = 1
options.enable_cpu_mem_arena = False

# Global sky segmentation session (will be initialized on first use)
skyseg_session = None
io_binding = None


class SkySegmentationProcessor:
    """Handles sky segmentation using ONNX model."""

    def __init__(self, model_path: str = "skyseg.onnx"):
        """Initialize the sky segmentation processor."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_path = model_path
        self.session = None
        self.io_binding = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the ONNX model and session."""
        if not os.path.exists(self.model_path):
            self._download_model()

        global options
        self.session = onnxruntime.InferenceSession(
            self.model_path,
            sess_options=options,
            providers=["CUDAExecutionProvider"]
        )
        self.io_binding = self.session.io_binding()

    def _download_model(self):
        """Download the sky segmentation model."""
        url = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx"
        self.logger.info("Downloading skyseg.onnx...")
        self._download_file_from_url(url, self.model_path)

    def _download_file_from_url(self, url: str, filename: str):
        """Download a file from URL with redirect handling."""
        try:
            response = requests.get(url, allow_redirects=False)
            response.raise_for_status()

            if response.status_code == 302:
                redirect_url = response.headers["Location"]
                response = requests.get(redirect_url, stream=True)
                response.raise_for_status()
            else:
                self.logger.info(f"Unexpected status code: {response.status_code}")
                return

            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            self.logger.info(f"Downloaded {filename} successfully.")

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Error downloading file: {e}")

    def segment_sky(self, image_path: str, mask_filename: Optional[str] = None) -> np.ndarray:
        """
        Segment sky from an image.

        Args:
            image_path: Path to input image
            mask_filename: Path to save the output mask

        Returns:
            Binary mask where 255 indicates non-sky regions
        """
        if mask_filename is None:
            raise ValueError("mask_filename must be provided")

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        result_map = self._run_inference([320, 320], image)
        result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

        output_mask = np.zeros_like(result_map_original)
        output_mask[result_map_original < 1] = 1
        output_mask = output_mask.astype(np.uint8) * 255

        os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
        cv2.imwrite(mask_filename, output_mask)
        return output_mask

    def _run_inference(self, input_size: List[int], image: np.ndarray) -> np.ndarray:
        """Run sky segmentation inference."""
        # Preprocess image
        temp_image = copy.deepcopy(image)
        resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
        x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
        x = np.array(x, dtype=np.float32)

        # Normalize
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        x = (x / 255 - mean) / std
        x = x.transpose(2, 0, 1)
        x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

        # Run inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        self.io_binding.bind_cpu_input(input_name, x)
        self.io_binding.bind_output(output_name)
        self.session.run_with_iobinding(self.io_binding)
        onnx_result = self.io_binding.copy_outputs_to_cpu()[0]

        # Post-process
        onnx_result = np.array(onnx_result).squeeze()
        min_value = np.min(onnx_result)
        max_value = np.max(onnx_result)
        onnx_result = (onnx_result - min_value) / (max_value - min_value)
        onnx_result *= 255
        onnx_result = onnx_result.astype("uint8")

        return onnx_result


class PointCloudProcessor:
    """Handles point cloud processing and filtering."""

    def __init__(self):
        """Initialize the point cloud processor."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sky_processor = SkySegmentationProcessor()

    def filter_predictions(self, predictions: Dict, conf_thres: float = 50.0,
                          filter_by_frames: str = "all", mask_black_bg: bool = False,
                          mask_white_bg: bool = False, mask_sky: bool = False,
                          target_dir: Optional[str] = None, image_subdir: Optional[str] = None,
                          prediction_mode: str = "Predicted Pointmap",
                          only_render_last_24_frame: bool = False) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Filter and process predictions to extract valid 3D points.

        Returns:
            Tuple of (vertices_3d, colors_rgb, scene_scale)
        """
        # Extract point data
        pred_world_points, pred_world_points_conf = self._extract_point_data(
            predictions, prediction_mode
        )

        # Apply sky masking if requested
        if mask_sky and target_dir and image_subdir:
            pred_world_points_conf = self._apply_sky_mask(
                pred_world_points_conf, target_dir, image_subdir, only_render_last_24_frame
            )

        # Filter by frame selection
        if filter_by_frames != "all" and filter_by_frames != "All":
            selected_frame_idx = self._parse_frame_filter(filter_by_frames)
            if selected_frame_idx is not None:
                pred_world_points = pred_world_points[selected_frame_idx:selected_frame_idx+1]
                pred_world_points_conf = pred_world_points_conf[selected_frame_idx:selected_frame_idx+1]
                predictions["images"] = predictions["images"][selected_frame_idx:selected_frame_idx+1]
                predictions["extrinsic"] = predictions["extrinsic"][selected_frame_idx:selected_frame_idx+1]

        # Extract colors
        colors_rgb = self._extract_colors(predictions["images"])

        # Apply confidence filtering
        vertices_3d, colors_rgb = self._apply_confidence_filter(
            pred_world_points, pred_world_points_conf, colors_rgb, conf_thres
        )

        # Apply background masking
        if mask_black_bg or mask_white_bg:
            vertices_3d, colors_rgb = self._apply_background_mask(
                vertices_3d, colors_rgb, mask_black_bg, mask_white_bg
            )

        # Calculate scene scale
        scene_scale = self._calculate_scene_scale(vertices_3d)

        return vertices_3d, colors_rgb, scene_scale

    def _extract_point_data(self, predictions: Dict, prediction_mode: str) -> Tuple[np.ndarray, np.ndarray]:
        """Extract world points and confidence from predictions."""
        if "Pointmap" in prediction_mode:
            if "world_points" in predictions:
                pred_world_points = predictions["world_points"]
                pred_world_points_conf = predictions.get(
                    "world_points_conf",
                    np.ones_like(pred_world_points[..., 0])
                )
            else:
                self.logger.info("Warning: world_points not found, falling back to depth-based points")
                pred_world_points = predictions["world_points_from_depth"]
                pred_world_points_conf = predictions.get(
                    "depth_conf",
                    np.ones_like(pred_world_points[..., 0])
                )
        else:
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get(
                "depth_conf",
                np.ones_like(pred_world_points[..., 0])
            )

        return pred_world_points, pred_world_points_conf

    def _apply_sky_mask(self, conf: np.ndarray, target_dir: str, image_subdir: str,
                       only_render_last_24_frame: bool) -> np.ndarray:
        """Apply sky segmentation mask to confidence scores."""
        target_dir_images = os.path.join(target_dir, image_subdir)
        image_list = sorted(os.listdir(target_dir_images))

        if only_render_last_24_frame:
            image_list = image_list[:-24]

        sky_mask_list = []
        S, H, W = conf.shape

        for image_name in image_list:
            image_filepath = os.path.join(target_dir_images, image_name)
            mask_filepath = os.path.join(target_dir, "sky_masks", image_name)

            if os.path.exists(mask_filepath):
                sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
            else:
                sky_mask = self.sky_processor.segment_sky(image_filepath, mask_filepath)

            if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
                sky_mask = cv2.resize(sky_mask, (W, H))

            sky_mask_list.append(sky_mask)

        sky_mask_array = np.array(sky_mask_list)
        sky_mask_binary = (sky_mask_array > 0.01).astype(np.float32)
        return conf * sky_mask_binary

    def _parse_frame_filter(self, filter_by_frames: str) -> Optional[int]:
        """Parse frame filter specification."""
        try:
            return int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            return None

    def _extract_colors(self, images: np.ndarray) -> np.ndarray:
        """Extract RGB colors from images."""
        if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
            colors_rgb = np.transpose(images, (0, 2, 3, 1))
        else:  # NHWC format
            colors_rgb = images
        return (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    def _apply_confidence_filter(self, points: np.ndarray, conf: np.ndarray,
                               colors: np.ndarray, conf_thres: float) -> Tuple[np.ndarray, np.ndarray]:
        """Apply confidence threshold filtering."""
        conf_flat = conf.reshape(-1)

        if conf_thres == 0.0:
            conf_threshold = 0.0
        else:
            conf_threshold = np.percentile(conf_flat, conf_thres)

        conf_mask = conf_flat >= conf_threshold

        if not np.any(conf_mask):
            # Fallback if no points pass threshold
            return np.array([[1, 0, 0]]), np.array([[255, 255, 255]])

        return points.reshape(-1, 3)[conf_mask], colors[conf_mask]

    def _apply_background_mask(self, vertices: np.ndarray, colors: np.ndarray,
                             mask_black_bg: bool, mask_white_bg: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Apply background color masking."""
        conf_mask = np.ones(len(vertices), dtype=bool)

        if mask_black_bg:
            black_bg_mask = colors.sum(axis=1) >= 16
            conf_mask = conf_mask & black_bg_mask

        if mask_white_bg:
            white_bg_mask = ~((colors[:, 0] > 240) & (colors[:, 1] > 240) & (colors[:, 2] > 240))
            conf_mask = conf_mask & white_bg_mask

        if not np.any(conf_mask):
            return np.array([[1, 0, 0]]), np.array([[255, 255, 255]])

        return vertices[conf_mask], colors[conf_mask]

    def _calculate_scene_scale(self, vertices: np.ndarray) -> float:
        """Calculate scene scale based on point cloud extent."""
        if vertices is None or len(vertices) == 0:
            return 1.0

        lower_percentile = np.percentile(vertices, 5, axis=0)
        upper_percentile = np.percentile(vertices, 95, axis=0)
        return np.linalg.norm(upper_percentile - lower_percentile)

class SceneBuilder:
   
    
    """Handles 3D scene construction and camera integration."""

    def __init__(self):
        """Initialize the scene builder."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    def build_scene(self, vertices: np.ndarray, colors: np.ndarray,
                   camera_matrices: np.ndarray, scene_scale: float,
                   show_cam: bool = True) -> trimesh.Scene:
        """Build a 3D scene with point cloud and cameras."""
        scene_3d = trimesh.Scene()

        # Add point cloud
        point_cloud = trimesh.PointCloud(vertices=vertices, colors=colors)
        scene_3d.add_geometry(point_cloud)

        if show_cam:
            self._add_cameras_to_scene(scene_3d, camera_matrices, scene_scale)

        return scene_3d

    def _add_cameras_to_scene(self, scene: trimesh.Scene, camera_matrices: np.ndarray,
                            scene_scale: float):
        """Add camera visualizations to the scene."""
        # Convert to 4x4 matrices
        num_cameras = len(camera_matrices)
        extrinsics_matrices = np.zeros((num_cameras, 4, 4))
        extrinsics_matrices[:, :3, :4] = camera_matrices
        extrinsics_matrices[:, 3, 3] = 1

        for i in range(num_cameras):
            world_to_camera = extrinsics_matrices[i]
            camera_to_world = np.linalg.inv(world_to_camera)
            rgba_color = self.colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])

            self._integrate_camera_into_scene(scene, camera_to_world, current_color, scene_scale)

    def _integrate_camera_into_scene(self, scene: trimesh.Scene, transform: np.ndarray,
                                   face_colors: tuple, scene_scale: float):
        """Integrate a camera mesh into the scene."""
        cam_width = scene_scale * 0.05
        cam_height = scene_scale * 0.1

        # Create cone shape for camera
        rot_45_degree = np.eye(4)
        rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
        rot_45_degree[2, 3] = -cam_height

        opengl_transform = self._get_opengl_conversion_matrix()
        complete_transform = transform @ opengl_transform @ rot_45_degree

        camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

        # Generate mesh
        slight_rotation = np.eye(4)
        slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

        vertices_combined = np.concatenate([
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            self._transform_points(slight_rotation, camera_cone_shape.vertices),
        ])
        vertices_transformed = self._transform_points(complete_transform, vertices_combined)

        mesh_faces = self._compute_camera_faces(camera_cone_shape)

        # Add to scene
        camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
        camera_mesh.visual.face_colors[:, :3] = face_colors
        scene.add_geometry(camera_mesh)

    def _get_opengl_conversion_matrix(self) -> np.ndarray:
        """Get OpenGL conversion matrix."""
        matrix = np.identity(4)
        matrix[1, 1] = -1
        matrix[2, 2] = -1
        return matrix

    def _transform_points(self, transformation: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Apply transformation to points."""
        points = np.asarray(points)
        initial_shape = points.shape[:-1]
        dim = points.shape[-1]

        transformation = transformation.swapaxes(-1, -2)
        points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]
        result = points[..., :dim].reshape(*initial_shape, dim)
        return result

    def _compute_camera_faces(self, cone_shape: trimesh.Trimesh) -> np.ndarray:
        """Compute faces for camera mesh."""
        faces_list = []
        num_vertices_cone = len(cone_shape.vertices)

        for face in cone_shape.faces:
            if 0 in face:
                continue
            v1, v2, v3 = face
            v1_offset, v2_offset, v3_offset = face + num_vertices_cone
            v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone

            faces_list.extend([
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ])

        faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
        return np.array(faces_list)
    
    def build_open3d_scene(self, vertices: np.ndarray, colors: np.ndarray) -> o3d.visualization.rendering.OffscreenRenderer:
        renderer = o3d.visualization.rendering.OffscreenRenderer(512, 512)
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultUnlit"
        material.point_size = 1.0
        material.base_roughness = 1.0
        material.base_reflectance = 0.0
        scene = renderer.scene
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
        scene.add_geometry("pcd", pcd, material)
        scene.set_background([0.0, 0.0, 0.0, 1.0])
        return renderer

    def align_extrinsics(self, camera_pose, predictions_extrinsic, num_target_view, outdir, only_render_last_24_frame):
        import torch
        import numpy as np
        # Prepare 4x4 matrices for camera extrinsics
        camera_matrices = predictions_extrinsic
        num_cameras = len(camera_matrices)
        extrinsics_matrices = np.zeros((num_cameras, 4, 4))
        extrinsics_matrices[:, :3, :4] = camera_matrices
        extrinsics_matrices[:, 3, 3] = 1
        extrinsics_matrices_inv = []
        for extrinsics_matrix in extrinsics_matrices:
            extrinsics_matrix_inv = np.linalg.inv(extrinsics_matrix)
            extrinsics_matrices_inv.append(extrinsics_matrix_inv)
        extrinsics_matrices_inv = np.stack(extrinsics_matrices_inv)

        try:
            segment_id = int(outdir.rstrip("/").split("_")[-1])
        except:
            segment_id = 1
        if not only_render_last_24_frame:
            target_start_id = (segment_id + 1) * num_target_view + 1
        else:
            target_start_id = -num_target_view

        extrinsics_matrices_gt = camera_pose[:target_start_id]
        if not only_render_last_24_frame:
            target_extrinsic_gt = camera_pose[target_start_id : target_start_id + num_target_view]
        else:
            target_extrinsic_gt = camera_pose[target_start_id:]

        extrinsics_matrices_gt_t = torch.tensor(extrinsics_matrices_gt).cuda()
        extrinsics_matrices_inv_t = torch.tensor(extrinsics_matrices_inv).cuda()
        s, R, t = align_first_and_last_points(
            extrinsics_matrices_gt_t.cpu().numpy()[:, :3, 3],
            extrinsics_matrices_inv_t.cpu().numpy()[:, :3, 3],
        )
        transform_mat = np.eye(4)
        transform_mat[:3, :3] = s * R
        transform_mat[:3, 3] = t

        num_target_cams = len(target_extrinsic_gt)
        if isinstance(target_extrinsic_gt, torch.Tensor):
            target_extrinsic_gt = target_extrinsic_gt.cpu().numpy()
        new_target_extrinsic = np.zeros((num_target_cams, 4, 4))
        new_target_extrinsic[:, :4, :4] = np.einsum(
            "ij, bjk -> bik", transform_mat, target_extrinsic_gt
        )
        return new_target_extrinsic
    
    def remove_opend3d_scene(self, renderer):
        """Delete or cleanup the Open3D OffscreenRenderer and its scene."""
        # renderer: o3d.visualization.rendering.OffscreenRenderer
        scene = renderer.scene
        # Remove all geometries from the scene
        scene.clear_geometry()
        # Optionally reset background
        import gc
        gc.collect()


class CubemapRenderer:
    
    """Handles cubemap rendering and equirectangular conversion."""

    def __init__(self):
        """Initialize the cubemap renderer."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.start = True


    def cube_to_equirectangular_cuda(self,cube_faces_batch, width, height, device='cuda'):
        """
        Converts a batch of cube map faces to equirectangular panoramas using PyTorch and CUDA.
        """
        batch_size = cube_faces_batch['front'].shape[0]

        # Create grid for equirectangular coordinates
        x = torch.linspace(0, width - 1, width, device=device)
        y = torch.linspace(0, height - 1, height, device=device)
        xv, yv = torch.meshgrid(y, x, indexing='ij')  # Switch order for (height, width)

        xv = xv.unsqueeze(0).repeat(batch_size, 1, 1)
        yv = yv.unsqueeze(0).repeat(batch_size, 1, 1)

        lon = (-yv / width) * 2 * torch.pi - torch.pi + torch.pi / 2
        lat = (xv / height) * torch.pi - torch.pi / 2

        X = torch.cos(lat) * torch.cos(lon)
        Y = torch.sin(lat)
        Z = torch.cos(lat) * torch.sin(lon)

        absX = X.abs()
        absY = Y.abs()
        absZ = Z.abs()

        panoramas = torch.zeros((batch_size, height, width, 3), dtype=torch.uint8, device=device)

        face = torch.empty((batch_size, height, width), dtype=torch.int8, device=device)
        u = torch.zeros_like(X)
        v = torch.zeros_like(Y)
        masks = {
            'right': (absX >= absY) & (absX >= absZ) & (X > 0),
            'left': (absX >= absY) & (absX >= absZ) & (X < 0),
            'bottom': (absY >= absX) & (absY >= absZ) & (Y > 0),
            'top': (absY >= absX) & (absY >= absZ) & (Y < 0),
            'front': (absZ >= absX) & (absZ >= absY) & (Z > 0),
            'back': (absZ >= absX) & (absZ >= absY) & (Z < 0),
        }

        face_values = {
            'right': 0,
            'left': 1,
            'bottom': 2,
            'top': 3,
            'front': 4,
            'back': 5,
        }

        for f, mask in masks.items():
            face[mask] = face_values[f]
            if f in ['right', 'left']:
                u[mask] = -Z[mask] / absX[mask] if f == 'right' else Z[mask] / absX[mask]
                v[mask] = -Y[mask] / absX[mask]
            elif f in ['bottom', 'top']:
                u[mask] = -X[mask] / absY[mask]
                v[mask] = -Z[mask] / absY[mask] if f == 'bottom' else Z[mask] / absY[mask]
            elif f in ['front', 'back']:
                u[mask] = X[mask] / absZ[mask] if f == 'front' else -X[mask] / absZ[mask]
                v[mask] = -Y[mask] / absZ[mask]

        u = (u + 1) / 2
        v = (v + 1) / 2
        for f, face_idx in face_values.items():
            mask = (face == face_idx)
            u_px = (u * (cube_faces_batch[f].shape[3] - 1)).long()
            v_px = ((1 - v) * (cube_faces_batch[f].shape[2] - 1)).long()
            # Corrected indexing: Get the color values for each pixel
            batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1)
            sampled_face = cube_faces_batch[f][batch_indices, :, v_px, u_px]  # Shape: (batch_size, 3, height, width)
            # Mask must match spatial dimensions (batch_size, height, width)
            panoramas[mask] = sampled_face[mask]

        return panoramas.cpu().numpy()  # Convert back to NumPy
    
    
    def render_face(self,scene_3d, cam, fov=90, res=(512,512), outdir="demo_trimesh_render.png", do_flip=False, savefig=False):
        camera_pose = cam
        F_y = np.eye(4)
        F_y[:3, :3] = Rotation.from_euler("z", 180, degrees=True).as_matrix()
        if do_flip:
            camera_pose = camera_pose @ F_y
        extrinsic_c2w = np.linalg.inv(camera_pose)
        fx = res[0] / (2 * np.tan(np.radians(fov) / 2))
        fy = res[1] / (2 * np.tan(np.radians(fov) / 2))
        cx = res[0] / 2
        cy = res[1] / 2
        intrinsic = o3d.camera.PinholeCameraIntrinsic(res[0], res[1], fx, fy, cx, cy)
        scene_3d.setup_camera(intrinsic, extrinsic_c2w)
        img = scene_3d.render_to_image()
        if savefig:
            cv2.imwrite(outdir, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        return img

    def render_cubemap(self, scene_3d, cam, res=(512,512), outdir="demo_trimesh_render.png", savefig=False,
                        initial_transformation=None):
        
        cubemap = {
            "front": None,
            "left": None,
            "back": None,
            "right": None,
            "bottom": None,
            "top": None,
        }
        opengl_transform = get_opengl_conversion_matrix()
        # cam = np.linalg.inv(cam) #@ opengl_transform
        # os.makedirs(outdir, exist_ok=True)
        for face, transform in CUBEMAP.items():
            output_path = os.path.join(outdir, f"{face}.png")
            if face in ['top','bottom']:
                do_flip = True
            else:
                do_flip = False
            img = self.render_face(
                scene_3d,
                cam  @ transform,
                res=res,
                outdir=output_path,
                do_flip=do_flip,
                savefig=savefig,

            )
            cubemap[face] = img
        return cubemap
    
    def render_cubemaps_to_panoramas(
        self,
        scene_3d: trimesh.Scene,
        target_extrinsic: np.ndarray,
        orig_extrinsic: np.ndarray,
        num_target_view: int = 24,
        outdir: str = "demo_pyrender_render",
        only_render_last_24_frame: bool = False,
    ):
        """
        Render cubemaps for each target extrinsic and convert to panoramas, saving the results.
        Args:
            scene_3d: The 3D scene to render.
            target_extrinsic: Array of aligned target camera extrinsics (N, 4, 4).
            orig_extrinsic: Original camera extrinsics (unused, for API compatibility).
            num_target_view: Number of target views to render.
            outdir: Output directory for panoramas.
            only_render_last_24_frame: Whether to only render the last 24 frames.
        """
        import torch
        import cv2
        import os
        import numpy as np

        cubemaps = {face: [] for face in CUBEMAP.keys()}
        os.makedirs(outdir, exist_ok=True)
        # Render cubemap for each target extrinsic
        for idx, render_extrinsic in enumerate(target_extrinsic):
            cubemap = self.render_cubemap(scene_3d, render_extrinsic, savefig=False)
            for face in cubemap:
                cubemaps[face].append(cubemap[face])
        # Convert cubemap lists to tensors
        for face in cubemaps:
            cubemaps[face] = (
                torch.from_numpy(np.stack(cubemaps[face])).to("cuda").permute(0, 3, 1, 2)
            )
        # Convert cubemaps to panoramas
        panoramas = self.cube_to_equirectangular_cuda(cubemaps, 2000, 1000)
        # Save panoramas
        for idx, panorama in enumerate(panoramas):
            cv2.imwrite(
                os.path.join(outdir, f"{idx:02}.png"), cv2.cvtColor(panorama, cv2.COLOR_RGB2BGR)
            )
        return panoramas

def predictions_to_glb(
    predictions: Dict,
    conf_thres: float = 50.0,
    filter_by_frames: str = "all",
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    mask_sky: bool = False,
    target_dir: Optional[str] = None,
    image_subdir: Optional[str] = None,
    prediction_mode: str = "Predicted Pointmap",
    only_render_last_24_frame: bool = False
) -> trimesh.Scene:
    """
    Convert VGGT predictions to a 3D scene represented as a GLB file.

    Args:
        predictions: Dictionary containing model predictions
        conf_thres: Confidence threshold percentage (default: 50.0)
        filter_by_frames: Frame filter specification (default: "all")
        mask_black_bg: Whether to mask black background (default: False)
        mask_white_bg: Whether to mask white background (default: False)
        show_cam: Whether to show cameras in scene (default: True)
        mask_sky: Whether to apply sky segmentation (default: False)
        target_dir: Directory containing images for sky masking
        image_subdir: Subdirectory with images
        prediction_mode: Prediction mode selector
        only_render_last_24_frame: Whether to exclude last 24 frames

    Returns:
        trimesh.Scene: Processed 3D scene

    Raises:
        ValueError: If predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    # Initialize processors
    point_processor = PointCloudProcessor()
    scene_builder = SceneBuilder()

    # Process and filter predictions
    vertices_3d, colors_rgb, scene_scale = point_processor.filter_predictions(
        predictions, conf_thres, filter_by_frames, mask_black_bg, mask_white_bg,
        mask_sky, target_dir, image_subdir, prediction_mode, only_render_last_24_frame
    )

    # Build scene
    scene_3d = scene_builder.build_scene(
        vertices_3d, colors_rgb, predictions["extrinsic"], scene_scale, show_cam
    )

    return scene_3d


def integrate_camera_into_scene(
    scene: trimesh.Scene,
    transform: np.ndarray,
    face_colors: tuple,
    scene_scale: float,
):
    """
    Integrates a fake camera mesh into the 3D scene.

    Args:
        scene (trimesh.Scene): The 3D scene to add the camera model.
        transform (np.ndarray): Transformation matrix for camera positioning.
        face_colors (tuple): Color of the camera face.
        scene_scale (float): Scale of the scene.
    """

    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    # Create cone shape for camera
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    # Combine transformations
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    # Generate mesh for the camera
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(complete_transform, vertices_combined)

    mesh_faces = compute_camera_faces(camera_cone_shape)

    # Add the camera mesh to the scene
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def apply_scene_alignment(scene_3d: trimesh.Scene, extrinsics_matrices: np.ndarray) -> trimesh.Scene:
    """
    Aligns the 3D scene based on the extrinsics of the first camera.

    Args:
        scene_3d (trimesh.Scene): The 3D scene to be aligned.
        extrinsics_matrices (np.ndarray): Camera extrinsic matrices.

    Returns:
        trimesh.Scene: Aligned 3D scene.
    """
    # Set transformations for scene alignment
    opengl_conversion_matrix = get_opengl_conversion_matrix()

    # Rotation matrix for alignment (180 degrees around the y-axis)
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()
    # Apply transformation
    initial_transformation = np.linalg.inv(extrinsics_matrices[0]) @ opengl_conversion_matrix
    for node in list(scene_3d.get_nodes()):
        if node.mesh is not None:
            # Get current transform
            current_transform = scene_3d.get_pose(node)
            # Apply new transform
            new_transform =initial_transformation @ current_transform
            scene_3d.set_pose(node, pose=new_transform)
    return scene_3d, initial_transformation



def get_opengl_conversion_matrix() -> np.ndarray:
    """
    Constructs and returns the OpenGL conversion matrix.

    Returns:
        numpy.ndarray: A 4x4 OpenGL conversion matrix.
    """
    # Create an identity matrix
    matrix = np.identity(4)

    # Flip the y and z axes
    matrix[1, 1] = -1
    matrix[2, 2] = -1

    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int = None) -> np.ndarray:
    """
    Applies a 4x4 transformation to a set of points.

    Args:
        transformation (np.ndarray): Transformation matrix.
        points (np.ndarray): Points to be transformed.
        dim (int, optional): Dimension for reshaping the result.

    Returns:
        np.ndarray: Transformed points.
    """
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    # Apply transformation
    transformation = transformation.swapaxes(-1, -2)  # Transpose the transformation matrix
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    # Reshape the result
    result = points[..., :dim].reshape(*initial_shape, dim)
    return result


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    """
    Computes the faces for the camera mesh.

    Args:
        cone_shape (trimesh.Trimesh): The shape of the camera cone.

    Returns:
        np.ndarray: Array of faces for the camera mesh.
    """
    # Create pseudo cameras
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)

    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone

        faces_list.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )

    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


def segment_sky(image_path, onnx_session, mask_filename=None):
    """
    Segments sky from an image using an ONNX model.

    Args:
        image_path: Path to input image
        onnx_session: ONNX runtime session with loaded model
        mask_filename: Path to save the output mask

    Returns:
        np.ndarray: Binary mask where 255 indicates non-sky regions
    """

    assert mask_filename is not None
    image = cv2.imread(image_path)

    result_map = run_skyseg(onnx_session, [320, 320], image)
    # resize the result_map to the original image size
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 1] = 1
    output_mask = output_mask.astype(np.uint8) * 255
    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    cv2.imwrite(mask_filename, output_mask)
    return output_mask


def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.

    Args:
        onnx_session: ONNX runtime session
        input_size: Target size for model input (width, height)
        image: Input image in BGR format

    Returns:
        np.ndarray: Segmentation mask
    """
    global io_binding
    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    io_binding.bind_cpu_input(input_name,x)
    io_binding.bind_output(output_name)
    onnx_session.run_with_iobinding(io_binding)
    onnx_result = io_binding.copy_outputs_to_cpu()[0]

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result




def get_camera_transformation(gt_cam, pred_cam):
    """
    Computes the scale (theta) and SE(3) transformation (F) from pred_cam to gt_cam using least squares.

    Args:
        gt_cam (np.ndarray): Ground truth camera extrinsic matrices of shape (N, 4, 4).
        pred_cam (np.ndarray): Predicted camera extrinsic matrices of shape (N, 4, 4).

    Returns:
        theta (float): Scale factor.
        F (np.ndarray): SE(3) transformation matrix (4x4).
    """
    assert gt_cam.shape == pred_cam.shape and gt_cam.shape[1:] == (4, 4), "Camera matrices must be (N, 4, 4)"
    
    N = gt_cam.shape[0]
    
    device = gt_cam.device
    gt_cam = gt_cam.to(torch.float64)
    
    gt_centers = torch.inverse(gt_cam)[:, :3, 3]
    pred_centers = torch.inverse(pred_cam)[:, :3, 3]
    
    # Step 1: Estimate scale θ via least squares
    num = torch.sum(gt_centers * pred_centers)
    denom = torch.sum(pred_centers ** 2)
    theta = num / denom
    
    pred_centers_scaled = theta * pred_centers
    
    # Step 2: Estimate rotation R and translation t via Kabsch algorithm
    centroid_gt = gt_centers.mean(dim=0)
    centroid_pred = pred_centers_scaled.mean(dim=0)
    
    pred_centered = pred_centers_scaled - centroid_pred
    gt_centered = gt_centers - centroid_gt
    
    H = pred_centered.T @ gt_centered
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    
    if torch.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    t = centroid_gt - R @ centroid_pred
    
    # Step 3: Construct SE(3) matrix F
    F = torch.eye(4, device=device)
    F[:3, :3] = R
    F[:3, 3] = t
    
    return theta, F

def transfrom_cam_to_gt(pred, F, theta):
    """
    Transforms predicted camera matrices to ground truth space using estimated SE(3) transformation and scale.

    Args:
        pred (torch.Tensor): Predicted camera extrinsic matrices of shape (N, 4, 4).
        F (torch.Tensor): SE(3) transformation matrix (4, 4).
        theta (torch.Tensor or float): Estimated scale factor.

    Returns:
        torch.Tensor: Transformed camera extrinsics in ground truth space (N, 4, 4).
    """
    device = pred.device
    pred = pred.to(torch.float64)
    F = F.to(torch.float64)

    # Get camera centers in world coordinates
    pred_centers = torch.inverse(pred)[:, :3, 3]  # (N, 3)

    # Apply scale
    pred_centers_scaled = theta * pred_centers  # (N, 3)

    # Apply SE(3) transformation: F[:3, :3] is rotation, F[:3, 3] is translation
    R = F[:3, :3]  # (3, 3)
    t = F[:3, 3]   # (3,)
    gt_centers = (R @ pred_centers_scaled.T).T + t  # (N, 3)

    # Get original camera orientations (in world-to-camera space)
    pred_rot = torch.inverse(pred)[:, :3, :3]  # (N, 3, 3)

    # Compose new camera-to-world matrix
    new_cam_to_world = torch.eye(4, dtype=torch.float64, device=device).unsqueeze(0).repeat(pred.shape[0], 1, 1)
    new_cam_to_world[:, :3, :3] = pred_rot
    new_cam_to_world[:, :3, 3] = gt_centers

    # Convert back to world-to-camera extrinsics
    new_world_to_cam = torch.inverse(new_cam_to_world)

    return new_world_to_cam


def plot_extrinsics(extrinsics_gt, extrinsics_pred, ax=None):
    """
    Plots the extrinsics of ground truth and predicted camera matrices.

    Args:
        extrinsics_gt (torch.Tensor): Ground truth camera extrinsics (N, 4, 4).
        extrinsics_pred (torch.Tensor): Predicted camera extrinsics (N, 4, 4).
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If None, a new figure is created.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    # Plot ground truth camera positions
    gt_centers = extrinsics_gt[:, :3, 3]
    ax.scatter(gt_centers[:, 0], gt_centers[:, 2], color='g', label='GT Cameras')
    ax.plot(gt_centers[:, 0], gt_centers[:, 2], color='g', alpha=0.5)

    # Plot predicted camera positions
    pred_centers = extrinsics_pred[:, :3, 3]
    ax.scatter(pred_centers[:, 0], pred_centers[:, 2], color='r', label='Pred Cameras',s=2)
    ax.plot(pred_centers[:, 0], pred_centers[:, 2], color='r', alpha=0.5)

    # ax.set_xlabel('X')
    # ax.set_zlabel('Z')
    ax.legend()
    # plt.show()
    # save as png
    plt.savefig("extrinsics_plot.png", dpi=300)


def rotation_from_vectors(u, v):
    """
    Return the 3x3 rotation matrix that rotates vector u onto vector v.
    Both u and v are 3D. 
    If u or v is near zero-length, or if they're nearly parallel/antiparallel,
    handle those as special cases.
    """
    # Normalize the input vectors:
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < 1e-15 or norm_v < 1e-15:
        # One is zero -> no well-defined rotation; return identity
        return np.eye(3)
    
    u_hat = u / norm_u
    v_hat = v / norm_v
    
    dot = np.dot(u_hat, v_hat)
    # Numerical safety:
    dot = np.clip(dot, -1.0, 1.0)
    
    if np.isclose(dot, 1.0):
        # Vectors are almost identical => no rotation needed
        return np.eye(3)
    elif np.isclose(dot, -1.0):
        # Vectors are opposite => rotate 180 deg around any axis perpendicular to u
        # e.g., find some vector orthonormal to u_hat
        # For instance, we can try to take cross of u_hat with [1,0,0] or [0,1,0], etc.
        # Then build a 180 rotation from that axis.
        temp = np.array([1.0, 0.0, 0.0])
        if np.abs(np.dot(u_hat, temp)) > 0.9:
            temp = np.array([0.0, 1.0, 0.0])
        w = np.cross(u_hat, temp)
        w /= np.linalg.norm(w)
        # Rotation by 180 deg around w => R = I - 2*w*w^T
        R = np.eye(3) - 2.0 * np.outer(w, w)
        return R
    else:
        # General case
        angle = np.arccos(dot)
        w = np.cross(u_hat, v_hat)
        w_norm = np.linalg.norm(w)
        w_hat = w / w_norm  # unit rotation axis
        # Rodrigues' rotation formula:
        K = np.array([[0,        -w_hat[2],  w_hat[1]],
                      [w_hat[2],  0,        -w_hat[0]],
                      [-w_hat[1], w_hat[0],  0       ]])
        R = np.eye(3) + np.sin(angle)*K + (1.0 - np.cos(angle))*(K @ K)
        return R

def align_first_and_last_points(A, B):
    """
    Given two sets of 3D points A and B (shape (N,3)), 
    compute scale s, rotation R, and translation t such that
       B0 = s*R*A0 + t
       B_{N-1} = s*R*A_{N-1} + t
    Returns (s, R, t).
    """
    # Extract the "anchor" points
    A0 = A[0]
    A1 = A[-1]
    B0 = B[0]
    B1 = B[-1]
    
    # Vectors from first to last
    vA = A1 - A0
    vB = B1 - B0
    
    # Scale
    lenA = np.linalg.norm(vA)
    lenB = np.linalg.norm(vB)
    if lenA < 1e-15:
        # Degenerate case: the first and last points of A are the same
        # => can't define a scale using these two points
        # We'll define scale = 1, rotation = I, translation so that A0 maps to B0
        s = 1.0
        R = np.eye(3)
        t = B0 - R @ A0
        return s, R, t
    s = lenB / lenA
    
    # Rotation that takes vA to vB
    R = rotation_from_vectors(vA, vB)
    
    # Translation
    t = B0 - s * R @ A0
    
    return s, R, t


def predictions_to_target_view(
    predictions: Dict,
    camera_pose: np.ndarray,
    conf_thres: float = 50.0,
    filter_by_frames: str = "all",
    point_processor=PointCloudProcessor(),
    scene_builder=SceneBuilder(),
    cubemap_renderer=CubemapRenderer(),
    mask_black_bg: bool = False,
    mask_white_bg: bool = False,
    show_cam: bool = True,
    mask_sky: bool = False,
    target_dir: Optional[str] = None,
    image_subdir: Optional[str] = None,
    prediction_mode: str = "Predicted Pointmap",
    num_target_view: int = 24,
    outdir: str = "demo_pyrender_render",
    only_render_last_24_frame: bool = False
) -> trimesh.Scene:
    """
    Convert VGGT predictions to a 3D scene and reproject to target views.

    Args:
        predictions: Dictionary containing model predictions
        camera_pose: Camera pose matrices for alignment
        conf_thres: Confidence threshold percentage (default: 50.0)
        filter_by_frames: Frame filter specification (default: "all")
        mask_black_bg: Whether to mask black background (default: False)
        mask_white_bg: Whether to mask white background (default: False)
        show_cam: Whether to show cameras in scene (default: True)
        mask_sky: Whether to apply sky segmentation (default: False)
        target_dir: Directory containing images for sky masking
        image_subdir: Subdirectory with images
        prediction_mode: Prediction mode selector
        num_target_view: Number of target views for reprojection
        outdir: Output directory for rendered images
        only_render_last_24_frame: Whether to exclude last 24 frames

    Returns:
        trimesh.Scene: Processed 3D scene

    Raises:
        ValueError: If predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    # Process and filter predictions
    vertices_3d, colors_rgb, scene_scale = point_processor.filter_predictions(
        predictions, conf_thres, filter_by_frames, mask_black_bg, mask_white_bg,
        mask_sky, target_dir, image_subdir, prediction_mode, only_render_last_24_frame
    )

    # Build Open3D scene using SceneBuilder
    scene_3d = scene_builder.build_open3d_scene(vertices_3d, colors_rgb)

    # Alignment step using SceneBuilder
    target_extrinsic = scene_builder.align_extrinsics(
        camera_pose, predictions["extrinsic"], num_target_view, outdir, only_render_last_24_frame
    )

    panoramas = cubemap_renderer.render_cubemaps_to_panoramas(
        scene_3d, target_extrinsic, predictions["extrinsic"], num_target_view,
        outdir, only_render_last_24_frame
    )
    scene_builder.remove_opend3d_scene(scene_3d)
    return panoramas



