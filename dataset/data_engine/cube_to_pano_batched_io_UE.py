import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
import torch
from multiprocessing import Pool


def load_cube_faces(face_paths):
    faces = {}
    for face in ['front', 'back', 'left', 'right', 'top', 'bottom']:
        face_path = face_paths[face]
        face_img = cv2.imread(face_path)
        if face in ["top","bottom"]:
            face_img = cv2.rotate(face_img,cv2.ROTATE_180)
        faces[face] = face_img
    return faces

def save_panorama(output_file, panorama):
    """Save a single panorama to disk."""
    cv2.imwrite(output_file, panorama)



def process_batch(cube_faces_paths_list, cube_ids_list, output_path, width, height):
    """
    Processes a batch of cube maps using GPU for conversion and multiprocess for I/O.
    """
    # Use multiprocessing to load cube faces in parallel
    with Pool() as pool:
        cube_faces_list = pool.map(load_cube_faces, cube_faces_paths_list)

    # Organize cube faces into batch format
    cube_faces_batch = {face: [] for face in ['front', 'back', 'left', 'right', 'top', 'bottom']}
    for cube_faces in cube_faces_list:
        for face in cube_faces.keys():
            cube_faces_batch[face].append(cube_faces[face])

    # Convert to PyTorch tensors
    for face in cube_faces_batch.keys():
        cube_faces_batch[face] = torch.from_numpy(np.stack(cube_faces_batch[face])).to('cuda').permute(0, 3, 1, 2)

    # Generate panoramas
    panoramas = cube_to_equirectangular_cuda(cube_faces_batch, width, height)

    # Save panoramas using multiprocessing
    save_args = [
        (os.path.join(output_path, f'{cube_ids_list[idx]+1:03}.png'), panoramas[idx])
        for idx in range(len(cube_faces_paths_list))
    ]
    num_processes=8
    with Pool(processes=num_processes) as pool:
        pool.starmap(save_panorama, save_args)


def cube_to_equirectangular_cuda(cube_faces_batch, width, height, device='cuda'):
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

def cube_to_equirectangular_cuda_UE(cube_faces_batch, width, height, device='cuda'):
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

    lon = (-yv / width) * 2 * torch.pi + torch.pi + torch.pi / 2
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


def read_and_convert_batch(cube_paths, output_path, width, height):
    """
    Reads and converts a batch of cube maps.
    cube_paths: List of paths to cube maps.
    """
    cube_faces_batch = {face: [] for face in ['front', 'back', 'left', 'right', 'top', 'bottom']}

    for cube_path in cube_paths:
        for face in cube_faces_batch.keys():
            face_path = os.path.join(cube_path, f"{face}.png")
            cube_faces_batch[face].append(cv2.imread(face_path))
    
    # Stack images for batch processing
    for face in cube_faces_batch.keys():
        cube_faces_batch[face] = np.stack(cube_faces_batch[face])

    for face in cube_faces_batch:
    # Convert NumPy array to PyTorch tensor and move to CUDA
        cube_faces_batch[face] = torch.from_numpy(cube_faces_batch[face]).to('cuda').permute(0, 3, 1, 2)

    # Generate panoramas for the batch
    panoramas = cube_to_equirectangular_cuda(cube_faces_batch, width, height)

    # Save panoramas
    for i, cube_path in enumerate(cube_paths):
        cube_id = os.path.basename(cube_path)
        output_file = os.path.join(output_path, f'{cube_id}.png')
        cv2.imwrite(output_file, panoramas[i])
        # print(f"Panorama saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Convert cube map to equirectangular panorama using broadcasting.')
    parser.add_argument('--image-path', type=str, required=True, help='Path to cube map images.')
    parser.add_argument('--output-path', type=str, required=True, help='Output directory for the panorama images.')
    parser.add_argument('--width', type=int, default=2000, help='Width of the output panorama.')
    parser.add_argument('--height', type=int, default=1000, help='Height of the output panorama.')
    parser.add_argument('--batch-size', type=int, default=50, help='Number of cube maps to process in a batch.')

    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    from collections import defaultdict
    import re

    id_to_faces = defaultdict(dict)
    pattern = re.compile(r'(\d+)_(top|bottom|left|right|front|back)\.png')

    for filename in os.listdir(args.image_path):
        match = pattern.match(filename)
        if match:
            id_str, face = match.groups()
            id_to_faces[int(id_str)][face] = os.path.join(args.image_path, filename)
    cube_ids = sorted(id_to_faces.keys())

    batch_size = args.batch_size
    for i in tqdm(range(0, len(cube_ids), batch_size)):
        batch_ids = cube_ids[i:i + batch_size]
        batch_faces = [id_to_faces[cube_id] for cube_id in batch_ids]
        process_batch(batch_faces, batch_ids, args.output_path, args.width, args.height)

if __name__ == "__main__":
    main()
