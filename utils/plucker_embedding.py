import numpy as np
import cv2

from scipy.ndimage import zoom

def equirectangular_to_ray_old(H, W, target_H=576, target_W=1024, up_axis='z'):
    """
    Compute 3D ray directions for each pixel in an equirectangular image
    and interpolate to match resized image dimensions.

    Parameters:
        H (int): Height of the original equirectangular image.
        W (int): Width of the original equirectangular image.
        target_H (int): Height of the resized image.
        target_W (int): Width of the resized image.

    Returns:
        np.ndarray: A (target_H, target_W, 3) array of 3D unit vectors
                    representing ray directions for the resized image.
    """
    # Create a grid of pixel coordinates for the original dimensions
    j, i = np.meshgrid(range(W), range(H), indexing='xy')

    # Convert pixel coordinates to normalized spherical coordinates
    # Longitude (phi): ranges from -pi to pi
    phi = (j / W) * 2 * np.pi - np.pi
    # Latitude (theta): ranges from -pi/2 to pi/2
    theta = (i / H) * np.pi - (np.pi / 2)

    # Convert spherical coordinates to Cartesian coordinates
    x = np.cos(theta) * np.cos(phi)
    y = np.cos(theta) * np.sin(phi)
    z = np.sin(theta)

    # Combine into a single array of shape (H, W, 3)
    rays = np.stack((x, y, z), axis=-1)

    # Normalize the rays (to ensure unit vectors)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    # Compute interpolation factors
    scale_h = target_H / H
    scale_w = target_W / W

    # Resize rays to match the resized image dimensions using interpolation
    rays_resized = zoom(rays, (scale_h, scale_w, 1), order=1)

    if  up_axis == 'y':
        rays_tempt =  rays_resized.copy()
        rays_resized[:,:,2] = - rays_tempt[:,:,1]
        rays_resized[:,:,1] = rays_tempt[:,:,2]
        rays_resized[:,:,0] = rays_tempt[:,:,0]

    return rays_resized

def equirectangular_to_ray(target_H=576, target_W=1024):
    """
    Compute 3D ray directions for each pixel in an equirectangular image
    and interpolate to match resized image dimensions, using OpenCV's RDF
    coordinate convention (X=right, Y=down, Z=forward).
    
    The center of the equirectangular image (W/2, H/2) corresponds to +Z.
    The top row corresponds to negative Y, the bottom row to positive Y,
    the left/right edges wrap around behind ±Z.

    Parameters:
        target_H (int): Height of the resized (output) image.
        target_W (int): Width of the resized (output) image.

    Returns:
        np.ndarray:
            A (target_H, target_W, 3) array of 3D unit vectors
            representing ray directions for the resized image.
    """
    # Create grid of pixel coordinates in the new (resized) domain
    # y' ranges [0, target_H-1], x' ranges [0, target_W-1]
    ys = np.arange(target_H, dtype=np.float32)
    xs = np.arange(target_W, dtype=np.float32)

    # Convert pixel coordinates to spherical angles (phi, theta).
    # phi   in [-π, π],    horizontal angle
    # theta in [-π/2, π/2], vertical angle
    # Center of image (x' = target_W/2, y' = target_H/2) => (phi=0, theta=0) => +Z
    phi   = (xs / target_W - 0.5) * 2.0 * np.pi   # shape (target_W,)
    theta = (ys / target_H - 0.5) * np.pi         # shape (target_H,)

    # Create a full 2D meshgrid so we get angle per (row,col)
    Phi, Theta = np.meshgrid(phi, theta)  # shape (target_H, target_W)

    # According to the chosen spherical-to-Cartesian mapping:
    #   X = cos(theta) * sin(phi)
    #   Y = sin(theta)
    #   Z = cos(theta) * cos(phi)
    #
    # This choice ensures:
    #  - center of image => +Z
    #  - top row => negative Y (since theta = -π/2 => sin(theta) = -1)
    #  - bottom row => positive Y
    #  - left and right edges => wrap around behind ±Z
    #
    # Also consistent with RDF: X is right, Y is down, Z is forward
    # (top of image is negative Y => up in a typical math sense, but "down" is +Y in RDF).

    cosT = np.cos(Theta)
    sinT = np.sin(Theta)
    sinP = np.sin(Phi)
    cosP = np.cos(Phi)

    dir_x = cosT * sinP
    dir_y = sinT
    dir_z = cosT * cosP

    # Stack into (H, W, 3). These are already unit vectors.
    directions = np.stack([dir_x, dir_y, dir_z], axis=-1)

    return directions


def equirectangular_to_ray_planar(target_H=576, target_W=1024):
    """
    Compute 3D ray directions for each pixel in an equirectangular image
    and interpolate to match resized image dimensions, using OpenCV's RDF
    coordinate convention (X=right, Y=down, Z=forward).
    
    The center of the equirectangular image (W/2, H/2) corresponds to +Z.
    The top row corresponds to negative Y, the bottom row to positive Y,
    the left/right edges wrap around behind ±Z.

    Parameters:
        target_H (int): Height of the resized (output) image.
        target_W (int): Width of the resized (output) image.

    Returns:
        np.ndarray:
            A (target_H, target_W, 3) array of 3D unit vectors
            representing ray directions for the resized image.
    """
    # Create grid of pixel coordinates in the new (resized) domain
    # y' ranges [0, target_H-1], x' ranges [0, target_W-1]
    ys = np.arange(target_H, dtype=np.float32)
    xs = np.arange(target_W, dtype=np.float32)

    # Convert pixel coordinates to spherical angles (phi, theta).
    # phi   in [-π/2, π/2],    horizontal angle
    # theta in [-arctan2, arctan2], vertical angle
    # Center of image (x' = target_W/2, y' = target_H/2) => (phi=0, theta=0) => +Z
    phi   = (xs / target_W - 0.5) * np.pi   # shape (target_W,)
    theta = (ys / target_H - 0.5) * 1.10714872 * 2  # shape (target_H,)

    # Create a full 2D meshgrid so we get angle per (row,col)
    Phi, Theta = np.meshgrid(phi, theta)  # shape (target_H, target_W)

    # According to the chosen spherical-to-Cartesian mapping:
    #   X = cos(theta) * sin(phi)
    #   Y = sin(theta)
    #   Z = cos(theta) * cos(phi)
    #
    # This choice ensures:
    #  - center of image => +Z
    #  - top row => negative Y (since theta = -π/2 => sin(theta) = -1)
    #  - bottom row => positive Y
    #  - left and right edges => wrap around behind ±Z
    #
    # Also consistent with RDF: X is right, Y is down, Z is forward
    # (top of image is negative Y => up in a typical math sense, but "down" is +Y in RDF).

    cosT = np.cos(Theta)
    sinT = np.sin(Theta)
    sinP = np.sin(Phi)
    cosP = np.cos(Phi)

    dir_x = cosT * sinP
    dir_y = sinT
    dir_z = cosT * cosP

    # Stack into (H, W, 3). These are already unit vectors.
    directions = np.stack([dir_x, dir_y, dir_z], axis=-1)

    return directions


def ray_c2w_to_plucker_np(ray, c2w):
    """
    Convert rays in camera coordinates to Plücker coordinates in world coordinates.

    Parameters:
        ray (np.ndarray): A (H, W, 3) array of 3D unit vectors representing ray directions.
        c2w (np.ndarray): A (N, 3, 4) array representing a sequence of camera-to-world transformation matrices.

    Returns:
        np.ndarray: A (N, 6, H, W) array of Plücker coordinates in world coordinates.
    """
    N, H, W, _ = c2w.shape[0], *ray.shape  # Number of transforms, and ray dimensions
    # Extract rotation matrices and translation vectors from c2w
    R = c2w[:, :3, :3]  # Shape (N, 3, 3)
    t = c2w[:, :3, 3]   # Shape (N, 3)
    

    # Convert ray directions to world coordinates for all c2w matrices
    rays_world = np.einsum('nij,hwj->nhwi', R, ray)  # Shape (N, H, W, 3)

    # Ray origins in world coordinates for all c2w matrices
    ray_origins_world = t[:, None, None, :]  # Shape (N, 1, 1, 3)

    # Compute Plücker coordinates
    # Moment (cross product of ray origins and directions)
    moment = np.cross(ray_origins_world, rays_world, axis=-1)  # Shape (N, H, W, 3)
    # Direction
    direction = rays_world  # Shape (N, H, W, 3)

    # Combine moment and direction into Plücker coordinates
    plucker = np.concatenate([moment, direction], axis=-1)  # Shape (N, H, W, 6)

    # Reorder to (N, 6, H, W)
    plucker = plucker.transpose(0, 3, 1, 2)  # Shape (N, 6, H, W)

    return plucker

import torch

def ray_c2w_to_plucker(ray, c2w):
    """
    Convert rays in camera coordinates to Plücker coordinates in world coordinates.

    Parameters:
        ray (torch.tensor): A (H, W, 3) tensor of 3D unit vectors representing ray directions.
        c2w (torch.tensor): A (N, 3, 4) tensor representing a sequence of camera-to-world transformation matrices.

    Returns:
        plucker (torch.tensor): A (N, 6, H, W) tensor of Plücker coordinates in world coordinates.
    """
    N, H, W, _ = c2w.shape[0], *ray.shape  # Number of transforms, and ray dimensions

    # Extract rotation matrices and translation vectors from c2w
    R = c2w[:, :3, :3]  # Shape (N, 3, 3)
    t = c2w[:, :3, 3]   # Shape (N, 3)

    # Convert ray directions to world coordinates for all c2w matrices
    rays_world = torch.einsum('nij,hwj->nhwi', R.float(), ray.float())  # Shape (N, H, W, 3)

    # Ray origins in world coordinates for all c2w matrices
    ray_origins_world = t[:, None, None, :]  # Shape (N, 1, 1, 3)
    ray_origins_world = ray_origins_world.float()
    rays_world = rays_world.float()
    # Compute Plücker coordinates
    # Moment (cross product of ray origins and directions)
    moment = torch.cross(ray_origins_world, rays_world, dim=-1)  # Shape (N, H, W, 3)

    # Combine moment and direction into Plücker coordinates
    plucker = torch.cat([rays_world, moment], dim=-1)  # Shape (N, H, W, 6)

    # Reorder to (N, 6, H, W)
    plucker = plucker.permute(0, 3, 1, 2)  # Shape (N, 6, H, W)

    return plucker



    
    

def image_to_point_cloud(image, downsample_factor=1):
    """
    Convert an equirectangular image into a colored spherical point cloud.

    Parameters:
        image (np.ndarray): A (H, W, 3) array representing the image (RGB values).
        downsample_factor (int): Factor by which to downsample the image. Defaults to 1 (no downsampling).

    Returns:
        np.ndarray: A (H' * W', 6) array where each row contains [x, y, z, R, G, B].
    """
    if downsample_factor > 1:
        # Downsample the image
        image = image[::downsample_factor, ::downsample_factor]

    H, W, _ = image.shape

    # Compute ray directions
    rays_normalized = equirectangular_to_ray(target_H=int(H/8), target_W=int(W/8))
    
    # Flatten the ray directions and image
    rays_flat = rays_normalized.reshape(-1, 3)
    colors_flat = image.reshape(-1, 3)

    # Combine ray directions and colors into a single point cloud
    point_cloud = np.hstack((rays_flat, colors_flat))

    return point_cloud

def rotate_point_cloud_x(point_cloud, angle_degrees):
    """
    Rotate the point cloud around the X-axis.

    Parameters:
        point_cloud (np.ndarray): A (N, 6) array where each row contains [x, y, z, R, G, B].
        angle_degrees (float): The rotation angle in degrees.

    Returns:
        np.ndarray: The rotated point cloud.
    """
    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)

    # Rotation matrix for X-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_radians), -np.sin(angle_radians)],
        [0, np.sin(angle_radians), np.cos(angle_radians)]
    ])

    # Rotate the points (first three columns of point_cloud)
    points = point_cloud[:, :3]
    rotated_points = points @ rotation_matrix.T

    # Combine rotated points with the original colors
    rotated_point_cloud = np.hstack((rotated_points, point_cloud[:, 3:]))

    return rotated_point_cloud

def save_point_cloud_to_obj(point_cloud, file_path):
    """
    Save the point cloud to an OBJ file.

    Parameters:
        point_cloud (np.ndarray): A (N, 6) array where each row contains [x, y, z, R, G, B].
        file_path (str): Path to save the OBJ file.
    """
    with open(file_path, 'w') as file:
        for point in point_cloud:
            x, y, z, r, g, b = point
            file.write(f"v {x} {y} {z} {r / 255.0} {g / 255.0} {b / 255.0}\n")

def save_point_cloud_to_ply(point_cloud, file_path):
    """
    Save the point cloud to a PLY file.

    Parameters:
        point_cloud (np.ndarray): A (N, 6) array where each row contains [x, y, z, R, G, B].
        file_path (str): Path to save the PLY file.
    """
    with open(file_path, 'w') as file:
        # Write PLY header
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {point_cloud.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")

        # Write point cloud data
        for point in point_cloud:
            x, y, z, r, g, b = point
            file.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")

def sample_random_se3(n=10, translation_range=1.0):
    """
    Generate a specified number of random SE(3) transformation matrices.

    Parameters:
        n (int): Number of random SE(3) transformation matrices to generate.
        translation_range (float): Range for the random translation components (uniformly sampled from -range to +range).

    Returns:
        np.ndarray: A (n, 4, 4) array of random SE(3) transformation matrices.
    """
    from scipy.spatial.transform import Rotation as R
    # Generate random rotations using quaternions
    random_quaternions = np.random.randn(n, 4)
    random_quaternions /= np.linalg.norm(random_quaternions, axis=1, keepdims=True)  # Normalize quaternions
    rotation_matrices = R.from_quat(random_quaternions).as_matrix()  # Convert to rotation matrices

    # Generate random translations
    translations = np.random.uniform(
        low=-translation_range, high=translation_range, size=(n, 3)
    )

    # Combine rotations and translations into SE(3) matrices
    se3_matrices = np.zeros((n, 4, 4))
    se3_matrices[:, :3, :3] = rotation_matrices  # Set rotation components
    se3_matrices[:, :3, 3] = translations  # Set translation components
    se3_matrices[:, 3, 3] = 1.0  # Homogeneous coordinate

    return se3_matrices