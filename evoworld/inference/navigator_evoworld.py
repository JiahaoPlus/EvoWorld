from tqdm import tqdm
import imageio
from evoworld.pipeline.pipeline_evoworld import StableVideoDiffusionPipeline
from utils.plucker_embedding import equirectangular_to_ray, ray_c2w_to_plucker
from evoworld.trainer.unet_plucker import UNetSpatioTemporalConditionModel
from torchvision import transforms
from PIL import Image
import numpy as np
from math import pi
import torch
import sys
from dataset.CameraTrajDataset import xyz_euler_to_three_by_four_matrix_batch

import logging

logging.basicConfig(
    level=logging.INFO,  # or DEBUG
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

all_close = lambda x, y: torch.allclose(x, y, atol=1e-5)

class CustomRescale:
    '''
    class: rescale the pxiel from [0,1] to [-1,1]
    '''
    def __init__(self):
        pass
    def __call__(self, img):
        return img * 2 - 1

def tensor_to_Image(tensor):
    '''
    Args:
        tensor: tensor with shape (N, 3, H, W)
    Returns:
        List[Image]: PIL image
    '''
    images = []
    if tensor.device.type == 'cuda':
        tensor = tensor.cpu()
    for i in range(tensor.size(0)):
        image = transforms.ToPILImage()(((tensor[i]/2 + 0.5)*255).to(torch.uint8))
        images.append(image)
    return images


class Navigator:
    def __init__(self, step_size=0.4,height=576,width=1024,position_scale=0.1):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.step_size = step_size
        self.position_scale = position_scale
        self.generations = []
        self.previous_images = torch.zeros(0,3,576,1024).to('cuda:0')
        self.previous_trajectoies = torch.tensor([]).to('cuda:0')
        self.current_pose = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.rays = torch.tensor(equirectangular_to_ray(target_H=int(height/8), target_W=int(width/8))).to(torch.float32).to('cuda:0')
        self.transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            CustomRescale()
        ])
        self.retrieve_time = 0
        self.start_image = None

    def set_current_image(self, image):
        self.generations.append([image])

    def get_current_image(self):
        return self.generations[-1][-1]

    def clear_movements(self):
        self.generations = []

    def get_all_frames(self):

        flattened_frames = [
            frame for movement in self.generations for frame in movement]
        return flattened_frames

    def get_current_trajectory(self):
        '''
        Get the current trajectory based on current pose following OPENCV RDF coordinate system
        '''
        first_pose = self.current_pose
        x, y, z, yaw, pitch, roll = first_pose
        # TODO: replaced sin and cos
        dir_x = np.sin(np.radians(pitch))
        dir_z = np.cos(np.radians(pitch))
        # current trajectory is 25 frames that has 0.4 interval on forwarding direction, starting from the first frame and keep the orientation
        current_trajectory = []
        current_trajectory.append([x, y, z, yaw, pitch, roll])
        for i in range(1,25):
            delta = i * self.step_size
            new_x = x + delta * dir_x
            new_y = y
            new_z = z + delta * dir_z
            current_trajectory.append([new_x, new_y, new_z, yaw, pitch, roll])
        current_trajectory = torch.tensor(current_trajectory)

        return current_trajectory

    def get_pipeline(self, unet_path, svd_path, num_frames=25, fps=7, progress_bar=True, image_width=1024, image_height=576, model_width=1024, model_height=None):
        unet = UNetSpatioTemporalConditionModel.from_pretrained(
            unet_path,
            subfolder="unet",
            # torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.logger.info('Unet Loaded')
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            svd_path,
            unet=unet,
            low_cpu_mem_usage=True,
            # torch_dtype=torch.float16, variant="fp16",
            local_files_only=True,
        )
        pipe.set_progress_bar_config(disable=not progress_bar)
        pipe.to("cuda:0")
        self.logger.info('Pipeline Loaded')
        self.pipe = pipe

        self.image_width = image_width
        self.image_height = image_height
        self.model_width = model_width
        self.model_height = model_height
        self.num_frames = num_frames
        self.fps = fps

        return pipe
        
    def extend_segment(self, segment, num_model_frames):
        """
        Extend the segment to the desired length.

        Parameters:
        segment (torch tensor): The segment to extend. [steps, 6] where 6 include x, y, z, rotx, roty, rotz
        num_model_frames (int): The desired length of the segment.
        Returns:
        torch tensor: The extended segment. If len(segment) < num_model_frames, extend the segment to num_model_frames.
        """
        
        if len(segment) == 0:
            return segment
        
        if len(segment) == 1:
            roty = segment[0][4]
            delta_z = self.step_size * torch.cos(torch.deg2rad(roty)) * self.position_scale
            delta_x = self.step_size * torch.sin(torch.deg2rad(roty)) * self.position_scale
            segment = torch.cat(segment).unsqueeze(0)
            for i in range(num_model_frames - 1):
                segment = torch.cat([segment, segment[-1].unsqueeze(0) + torch.tensor([delta_x, 0, delta_z, 0, 0, 0]).to(segment[0].device)], dim=0)

            return segment
        

        if len(segment) < num_model_frames:
            len_segment = len(segment)
            # extend using the same x y z interval, but keep rotation the same
            last_step = segment[-1]
            last_last_step = segment[-2]
            if not all_close(last_step[3:], last_last_step[3:]):
                from ipdb import set_trace; set_trace()
            assert all_close(last_step[3:], last_last_step[3:]), 'The rotation of the last two steps are not the same.'
            delta = last_step - last_last_step
            segment_extra = torch.zeros((num_model_frames - len_segment, 6)).to(segment[0].device)
            segment = torch.stack(segment)
            for i in range(num_model_frames - len_segment):
                segment_extra[i] = last_step + delta * (i+1)
            segment = torch.cat([segment, segment_extra], dim=0)
            return segment

    def move_forward(self, image=None, 
                    segment=None, num_model_frames=25, 
                    width=None, height=None, fps=None, 
                    num_inference_steps=50, noise_aug_strength=0.02,
                    use_memory=False):
        if image is None:
            image = self.generations[-1][-1]

        model_width = self.model_width if self.model_width else width
        model_height = self.model_height if self.model_height else width
        width = image.size(2) if not width else width
        height = image.size(1) if not height else height
        # image = self.transform(image).unsqueeze(0)
        num_frames = len(segment)
        if num_frames < num_model_frames:
            segment = self.extend_segment(segment, num_model_frames)

        camera_trajectory_raw = segment
        if not isinstance(camera_trajectory_raw, torch.Tensor):
            camera_trajectory_raw = torch.stack(camera_trajectory_raw, dim=0).to('cuda:0')
        all_cam_traj_raw = camera_trajectory_raw
        all_cam_traj = xyz_euler_to_three_by_four_matrix_batch(all_cam_traj_raw, relative=True, euler_as_rotation=False)
        all_plucker_embedding = ray_c2w_to_plucker(self.rays, all_cam_traj).to('cuda:0')
        plucker_embedding = all_plucker_embedding[:25].unsqueeze(0)
        memorized_plucker_embedding = all_plucker_embedding[25:].unsqueeze(0)
        generator = torch.manual_seed(-1)
        memorized_pixel_values = self.memorized_images.clone()
        with torch.inference_mode():
            frames = self.pipe(image.unsqueeze(0),
                               num_frames=self.num_frames,
                               width=model_width,
                               height=model_height,
                               decode_chunk_size=8, 
                               generator=generator, 
                               motion_bucket_id=127, 
                               fps=self.fps, 
                               num_inference_steps=num_inference_steps, 
                               noise_aug_strength=noise_aug_strength,
                               plucker_embedding=plucker_embedding,
                               memorized_pixel_values=memorized_pixel_values,
                               mask_mem=not use_memory).frames[0]
        self.logger.info(f'{num_frames} Frames Generated')
        frame_images = []
        for frame in frames:
            frame_image = self.transform(frame).unsqueeze(0)
            frame_images.append(frame_image)
        frame_images = torch.cat(frame_images, dim=0).to('cuda:0')
        self.previous_trajectoies = torch.cat([self.previous_trajectoies, camera_trajectory_raw],dim=0)
        self.previous_images = torch.cat([self.previous_images, frame_images], dim=0)
        # frames = tensor_to_Image(frames)
        frames = frames[:num_frames]
        frames = [frame.resize((width, height))
                  for frame in frames]

        self.generations.append(frames[1:])

        self.current_pose = camera_trajectory_raw[num_frames-1]

        return frames

    def save_video(self, save_path, fps=10):
        # Create a writer object
        writer = imageio.get_writer(save_path, fps=fps)

        for idx in range(len(self.generations)):
            if isinstance(self.generations[idx][0], torch.Tensor):
                self.generations[idx][0] = self.generations[idx][0] * 0.5 + 0.5
                self.generations[idx][0] = Image.fromarray((self.generations[idx][0]*255).detach().cpu().numpy().transpose(1, 2, 0).astype(np.uint8))
        frames = [frame.resize((self.image_width, self.image_height), Image.BICUBIC)
                  for movement in self.generations for frame in movement]

        if len(frames) == 0:
            self.logger.info('No Movement to Export.')
            return

        # Add images to the video
        for frame in tqdm(frames, desc="Processing Frames to Video"):
            # Convert the PIL image to a numpy array
            frame = np.array(frame)
            writer.append_data(frame)

        # Close the writer to finalize the video
        writer.close()

        self.logger.info(f'Video saved as {save_path}')

    def save_gif(self, save_path, fps=10):
        # Calculate the duration of each frame in the GIF
        duration = int(1000 / fps)  # duration in milliseconds per frame

        frames = [frame.resize((self.image_width, self.image_height), Image.BICUBIC)
                  for movement in self.generations for frame in movement]

        if len(frames) == 0:
            self.logger.info('No Movement to Export.')
            return

        # Convert frames to PIL images and save as GIF with duration
        frames[0].save(save_path, save_all=True,
                       append_images=frames[1:], duration=duration, loop=0)

        self.logger.info(f'GIF saved as {save_path}')

    def split_path_into_segments(self, path):
        """
        Split the path into segments based on the turn angle.
        Args:
            path: [steps, 6]
        """
        segments = []
        current_segment = []
        last_step = path[0]
        for step in path:
            if all_close(step[3:6], last_step[3:6]):
                current_segment.append(step.clone())
                last_step = step.clone()
                continue
            else:
                segments.append(current_segment)
                if len(segments) != 0:
                    last_step[3:6] = step[3:6]
                    current_segment = [last_step.clone()]
                    current_segment.append(step.clone())
                else:
                    current_segment = [step.clone()]
                last_step = step
        if current_segment:
            segments.append(current_segment)
        return segments
    
    def split_curve_into_segments(self, path):
        segments = []
        total_len = len(path)
        if total_len < 25:
            return [path]
        start_idx = 0
        end_idx = 25
        while end_idx <= total_len:
            segment = path[start_idx:end_idx]
            segments.append(segment)
            start_idx = end_idx - 1
            end_idx = start_idx + 25
        if end_idx - start_idx > 1 and start_idx < total_len:
            segment = path[start_idx:]
            segments.append(segment)
        return segments
    
    
    def get_idx_to_evaluate(self, segments):
        '''
        get the index to evaluate the loop consistency
        Args:
            segments: List[List[Tensor[6]]]
        Returns:
            int: the index to evaluate
        '''
        first_point = segments[0][0]
        all_points = torch.stack([point[0:3] for point in segments[-1]])
        distance_to_first = torch.norm(all_points - first_point[0:3], dim=1)
        idx = torch.argmin(distance_to_first)
        return idx

    def navigate_path(
        self, path, start_image, width=1024, height=512, fps=10, num_inference_steps=50,memorized_images=None,
        infer_segment=False, segment_id=None,
    ):
        """
        path: [steps, 6]
        """
        current_segment = 0
        self.memorized_images = memorized_images.clone()
        generations = []
        self.start_image = start_image

        current_image = start_image
        # generations.append([current_image])
        segments = self.split_path_into_segments(path)
        current_angle = segments[0][0][4]
        self.rotation_times = len(segments)
        self.total_distance = sum([len(segment) for segment in segments])
        while len(segments) > 0:
            segment = segments.pop(0)
            if segment_id is not None and current_segment < segment_id and infer_segment:
                current_segment += 1
                rotation_angle = segment[0][4] - current_angle
                current_angle = segment[-1][4]
                self.rotation_angle = rotation_angle
                continue
            if segment_id != 0:
                use_memory = True
            else:
                use_memory = False
            rotation_angle = segment[0][4] - current_angle
            current_angle = segment[-1][4]
            self.rotation_angle = rotation_angle
            if rotation_angle != 0:
                current_image = self.rotate_panorama(
                    current_image, rotation_angle, scale_factor=1)
            if len(segment) != 0:
                movement = self.move_forward(
                    current_image,
                    segment,
                    num_model_frames=25,
                    width=width,
                    height=height,
                    fps=fps,
                    num_inference_steps=num_inference_steps,
                    use_memory=use_memory
                )
                generations.append(movement)
                current_image = movement[-1]
                current_image = self.transform(current_image).to('cuda:0')
            else:
                generations.append([current_image])
            current_segment += 1
            if infer_segment and current_segment > segment_id:
                break

        self.generations = generations
        return generations
    
    def navigate_curve_path(
        self, path, start_image, width=1024, height=512, fps=10, num_inference_steps=50,memorized_images=None,
        infer_segment=False, segment_id=None,
    ):
        """
        path: [steps, 6]
        """
        current_segment = 0
        self.memorized_images = memorized_images.clone()
        generations = []
        self.start_image = start_image

        current_image = start_image
        # generations.append([current_image])
        segments = self.split_curve_into_segments(path)
        current_angle = segments[0][0][4]
        self.rotation_times = len(segments)
        self.total_distance = sum([len(segment) for segment in segments])
        while len(segments) > 0:
            segment = segments.pop(0)
            if segment_id is not None and current_segment < segment_id and infer_segment:
                current_segment += 1
                rotation_angle = segment[0][4] - current_angle
                current_angle = segment[-1][4]
                self.rotation_angle = rotation_angle
                continue
            if segment_id != 0:
                use_memory = True
            else:
                use_memory = False
            rotation_angle = segment[0][4] - current_angle
            current_angle = segment[-1][4]
            self.rotation_angle = rotation_angle
            if len(segment) != 0:
                movement = self.move_forward(
                    current_image,
                    segment,
                    num_model_frames=25,
                    width=width,
                    height=height,
                    fps=fps,
                    num_inference_steps=num_inference_steps,
                    use_memory=use_memory
                )
                generations.append(movement)
                current_image = movement[-1]
                current_image = self.transform(current_image).to('cuda:0')
            else:
                generations.append([current_image])
            current_segment += 1
            if infer_segment and current_segment > segment_id:
                break

        self.generations = generations
        return generations


    def get_rotation_times_and_total_distance(self,path):
        '''
        Args:
            path: List[Tensor[6]]
        Returns:
            int: the number of rotation times
            int: the total distance
        '''
        segments = self.split_path_into_segments(path)
        self.point_to_evaluate = self.get_idx_to_evaluate(segments)
        self.rotation_times = len(segments)
        segments[-1] = segments[-1][:self.point_to_evaluate+1]
        self.total_distance = sum([len(segment) for segment in segments])
        return self.rotation_times, self.total_distance

    def rotate_panorama(self, panorama_image=None, rotation_degrees=0, scale_factor=1):
        """
        Rotate an equirectangular panorama image along the z-axis using spherical coordinates.

        Parameters:
        panorama_image (torch.Tensor): The input equirectangular panorama image as a torch tensor.
        rotation_degrees (float): The amount to rotate the image in degrees. Positive values rotate to the right.
        scale_factor (int): The factor by which to scale the image for higher-quality processing.

        Returns:
        torch.Tensor: The rotated equirectangular panorama image.
        """
        if panorama_image is None:
            panorama_image = self.get_current_image()
        _, height, width = panorama_image.shape
        
        # Convert rotation degrees to radians (no negation for clockwise rotation)
        rotation_radians = torch.deg2rad(torch.tensor(rotation_degrees, dtype=torch.float32))
        
        # Create a meshgrid for the pixel coordinates
        x = torch.linspace(0, width - 1, width)
        y = torch.linspace(0, height - 1, height)
        xv, yv = torch.meshgrid(x, y, indexing='xy')
        
        # Calculate the spherical coordinates (longitude and latitude)
        longitude = (xv / width) * 2 * torch.pi
        latitude = (yv / height) * torch.pi - (torch.pi / 2)
        
        # Adjust the longitude by the rotation amount for clockwise rotation
        rotated_longitude = (longitude.to('cuda:0') + rotation_radians.to('cuda:0')) % (2 * torch.pi)
        
        # Convert spherical coordinates back to image coordinates
        uf = rotated_longitude / (2 * torch.pi) * width
        vf = (latitude + (torch.pi / 2)) / torch.pi * height
        
        # Ensure indices are within bounds
        ui = torch.clamp(uf, 0, width - 1).long().to(panorama_image.device)
        vi = torch.clamp(vf, 0, height - 1).long().to(panorama_image.device)
        
        # Create the rotated image using the new coordinates
        rotated_array = panorama_image[:,vi, ui]
        
        self.generations.append([rotated_array])
        
        self.current_pose[4] += rotation_degrees.to(self.current_pose.device)
        
        return rotated_array

    def convert_panorama_to_cubemap(self, panorama_image, interpolation=True, scale_factor=2):
        """
        Convert an equirectangular panorama image to a cube map with optional scaling.

        Parameters:
        panorama_image (Image): The input equirectangular panorama image.
        interpolation (bool): Whether to use bilinear interpolation for sampling.
        scale_factor (int): Factor by which to scale the input panorama.

        Returns:
        tuple: A tuple containing the cube map image and a dictionary of individual cube faces.
        """
        # Scale up the input panorama
        original_size = panorama_image.size
        scaled_size = (original_size[0] * scale_factor,
                       original_size[1] * scale_factor)
        panorama_image = panorama_image.resize(scaled_size, Image.LANCZOS)

        # Assert that the panorama image has the correct aspect ratio
        panorama_size = panorama_image.size
        assert panorama_size[0] == 2 * \
            panorama_size[1], "Panorama width must be twice the height."

        # Create an output image with appropriate dimensions for the cube map
        cubemap = Image.new(
            "RGB", (panorama_size[0], int(panorama_size[0] * 3 / 4)), "black")

        input_pixels = np.array(panorama_image)
        cubemap_pixels = np.zeros(
            (cubemap.size[1], cubemap.size[0], 3), dtype=np.uint8)  # Initialize with black
        edge_length = panorama_size[0] // 4  # Length of each edge in pixels

        # Create coordinate grids
        i, j = np.meshgrid(np.arange(cubemap.size[0]), np.arange(
            cubemap.size[1]), indexing='xy')

        # Assign face indices
        face_index = i // edge_length
        face_index[j < edge_length] = 4  # 'top'
        face_index[j >= 2 * edge_length] = 5  # 'bottom'

        def pixel_to_xyz(i, j, face):
            """
            Convert pixel coordinates of the output image to 3D coordinates based on the cube face index.

            Parameters:
            i (int): The x-coordinate of the pixel.
            j (int): The y-coordinate of the pixel.
            face (int): The face of the cube (0-5).

            Returns:
            tuple: A tuple (x, y, z) representing the 3D coordinates.
            """
            a = 2.0 * i / edge_length
            b = 2.0 * j / edge_length

            x = np.zeros_like(a, dtype=float)
            y = np.zeros_like(a, dtype=float)
            z = np.zeros_like(a, dtype=float)

            # Assign 3D coordinates based on the face index
            mask = face == 0  # back
            x[mask], y[mask], z[mask] = -1.0, 1.0 - a[mask], 3.0 - b[mask]

            mask = face == 1  # left
            x[mask], y[mask], z[mask] = a[mask] - 3.0, -1.0, 3.0 - b[mask]

            mask = face == 2  # front
            x[mask], y[mask], z[mask] = 1.0, a[mask] - 5.0, 3.0 - b[mask]

            mask = face == 3  # right
            x[mask], y[mask], z[mask] = 7.0 - a[mask], 1.0, 3.0 - b[mask]

            mask = face == 4  # top
            x[mask], y[mask], z[mask] = b[mask] - 1.0, a[mask] - 5.0, 1.0

            mask = face == 5  # bottom
            x[mask], y[mask], z[mask] = 5.0 - b[mask], a[mask] - 5.0, -1.0

            return x, y, z

        # Convert pixel coordinates to 3D coordinates
        x, y, z = pixel_to_xyz(i, j, face_index)
        theta = np.arctan2(y, x)  # Angle in the xy-plane
        r = np.hypot(x, y)  # Distance from origin in the xy-plane
        phi = np.arctan2(z, r)  # Angle from the z-axis

        # Source image coordinates
        uf = 2.0 * edge_length * (theta + pi) / pi
        vf = 2.0 * edge_length * (pi / 2 - phi) / pi

        if interpolation:
            # Bilinear interpolation
            ui = np.floor(uf).astype(int)
            vi = np.floor(vf).astype(int)
            u2 = ui + 1
            v2 = vi + 1
            mu = uf - ui
            nu = vf - vi

            # Ensure indices are within bounds
            ui = np.clip(ui, 0, panorama_size[0] - 1)
            vi = np.clip(vi, 0, panorama_size[1] - 1)
            u2 = np.clip(u2, 0, panorama_size[0] - 1)
            v2 = np.clip(v2, 0, panorama_size[1] - 1)

            # Pixel values of the four corners
            A = input_pixels[vi, ui]
            B = input_pixels[vi, u2]
            C = input_pixels[v2, ui]
            D = input_pixels[v2, u2]

            # Interpolate the RGB values
            R = A[:, :, 0] * (1 - mu) * (1 - nu) + B[:, :, 0] * mu * \
                (1 - nu) + C[:, :, 0] * (1 - mu) * nu + D[:, :, 0] * mu * nu
            G = A[:, :, 1] * (1 - mu) * (1 - nu) + B[:, :, 1] * mu * \
                (1 - nu) + C[:, :, 1] * (1 - mu) * nu + D[:, :, 1] * mu * nu
            B = A[:, :, 2] * (1 - mu) * (1 - nu) + B[:, :, 2] * mu * \
                (1 - nu) + C[:, :, 2] * (1 - mu) * nu + D[:, :, 2] * mu * nu

            interp_pixels = np.stack((R, G, B), axis=-1).astype(np.uint8)

            # Ensure all pure black pixels in the nearest-neighbor result are directly used in the final output
            cubemap_pixels = interp_pixels
        else:
            # Nearest-neighbor sampling
            ui = np.round(uf).astype(int)
            vi = np.round(vf).astype(int)

            valid = (ui >= 0) & (ui < panorama_size[0]) & (
                vi >= 0) & (vi < panorama_size[1])
            cubemap_pixels[(face_index >= 0) & valid] = input_pixels[vi[(
                face_index >= 0) & valid], ui[(face_index >= 0) & valid]]

        # First row: set empty spaces to black
        cubemap_pixels[0:edge_length, 0:edge_length] = [0, 0, 0]
        cubemap_pixels[0:edge_length, edge_length:2 * edge_length] = [0, 0, 0]
        cubemap_pixels[0:edge_length, 3 * edge_length:] = [0, 0, 0]

        # Third row: set empty spaces to black
        cubemap_pixels[2 * edge_length:3 *
                       edge_length, 0:edge_length] = [0, 0, 0]
        cubemap_pixels[2 * edge_length:3 * edge_length,
                       edge_length:2 * edge_length] = [0, 0, 0]
        cubemap_pixels[2 * edge_length:3 *
                       edge_length, 3 * edge_length:] = [0, 0, 0]

        # Convert the numpy array back to an image
        cubemap = Image.fromarray(cubemap_pixels)
        self.logger.info('Converted Panorama to Cube Map.')

        def extract_individual_faces():
            """
            Extract the individual cube faces from the cube map image.

            Returns:
            dict: A dictionary of individual cube faces with their names as keys.
            """
            # Define the names and coordinates for each face
            face_names = ['right', 'left', 'top', 'bottom', 'front', 'back']
            face_coordinates = [
                (edge_length * 3, edge_length,
                 edge_length * 4, edge_length * 2),  # right
                (edge_length, edge_length, edge_length * \
                 2, edge_length * 2),      # left
                (2 * edge_length, 0, 3 * edge_length,
                 edge_length),                # top
                (2 * edge_length, 2 * edge_length, 3 * \
                 edge_length, 3 * edge_length),  # bottom
                (2 * edge_length, edge_length, 3 * \
                 edge_length, edge_length * 2),  # front
                (0, edge_length, edge_length,
                 edge_length * 2)                     # back
            ]

            # Extract each face as an individual image and store it in a dictionary
            faces = {}
            for face_name, (x1, y1, x2, y2) in zip(face_names, face_coordinates):
                face_img = cubemap.crop((x1, y1, x2, y2))
                faces[face_name] = face_img
            return faces

        # Extract individual cube faces
        cubes = extract_individual_faces()
        self.logger.info('Extracted Cube Faces.')

        # Resize the cubemap back to its unscaled size
        unscaled_cubemap_size = (
            original_size[0], int(original_size[0] * 3 / 4))
        cubemap = cubemap.resize(unscaled_cubemap_size, Image.LANCZOS)

        return cubemap, cubes

    def precompute_rotation_matrix(self, rx, ry, rz):
        """
        Precompute a rotation matrix given rotation angles around x, y, and z axes.

        Parameters:
        rx, ry, rz: Rotation angles in degrees.

        Returns:
        numpy.ndarray: The resulting 3x3 rotation matrix.
        """
        # Convert degrees to radians
        rx = np.deg2rad(rx)
        ry = np.deg2rad(ry)
        rz = np.deg2rad(rz)

        # Rotation matrices for x, y, and z axes
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])

        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix
        R = Rz @ Ry @ Rx
        return R

    def cubemap_to_equirectangular(self, cubemap_faces, output_width, output_height, scale_factor=2):
        """
        Convert cube map images to a panorama image with optional scaling.

        Parameters:
        cubemap_faces (dict): Dictionary containing images of the cube map faces.
        output_width (int): Width of the output equirectangular image.
        output_height (int): Height of the output equirectangular image.
        scale_factor (int): Factor by which to scale the output equirectangular image for processing.

        Returns:
        Image: The equirectangular panorama image.
        """
        # Scale up the target resolution for processing
        scaled_output_width = output_width * scale_factor
        scaled_output_height = output_height * scale_factor

        # Precompute rotation matrix
        rx, ry, rz = 90, -90, 180  # Rotation parameters
        R = self.precompute_rotation_matrix(rx, ry, rz)

        # Create meshgrid for pixel coordinates
        x = np.linspace(0, scaled_output_width - 1, scaled_output_width)
        y = np.linspace(0, scaled_output_height - 1, scaled_output_height)
        xv, yv = np.meshgrid(x, y)

        # Convert equirectangular coordinates to spherical coordinates
        theta = (xv / scaled_output_width) * 2 * np.pi - np.pi
        phi = (yv / scaled_output_height) * np.pi - (np.pi / 2)

        # Convert spherical coordinates to Cartesian coordinates
        xs = np.cos(phi) * np.cos(theta)
        ys = np.cos(phi) * np.sin(theta)
        zs = np.sin(phi)

        def apply_rotation(x, y, z):
            """
            Apply a rotation matrix to 3D coordinates.

            Parameters:
            x, y, z: 3D coordinates as numpy arrays.

            Returns:
            numpy.ndarray: Rotated 3D coordinates.
            """
            return R @ np.array([x, y, z])

        # Apply precomputed rotation using the apply_rotation function
        xs, ys, zs = apply_rotation(xs.flatten(), ys.flatten(), zs.flatten())
        xs = xs.reshape((scaled_output_height, scaled_output_width))
        ys = ys.reshape((scaled_output_height, scaled_output_width))
        zs = zs.reshape((scaled_output_height, scaled_output_width))

        # Determine which face of the cubemap this point maps to
        abs_x, abs_y, abs_z = np.abs(xs), np.abs(ys), np.abs(zs)
        face_indices = np.argmax(
            np.stack([abs_x, abs_y, abs_z], axis=-1), axis=-1)

        equirectangular_pixels = np.zeros(
            (scaled_output_height, scaled_output_width, 3), dtype=np.uint8)

        for face_name, face_image in cubemap_faces.items():
            face_image = np.array(face_image)
            if face_name == 'right':
                mask = (face_indices == 0) & (xs > 0)
                u = (-zs[mask] / abs_x[mask] + 1) / 2
                v = (ys[mask] / abs_x[mask] + 1) / 2
            elif face_name == 'left':
                mask = (face_indices == 0) & (xs < 0)
                u = (zs[mask] / abs_x[mask] + 1) / 2
                v = (ys[mask] / abs_x[mask] + 1) / 2
            elif face_name == 'bottom':
                mask = (face_indices == 1) & (ys > 0)
                u = (xs[mask] / abs_y[mask] + 1) / 2
                v = (-zs[mask] / abs_y[mask] + 1) / 2
            elif face_name == 'top':
                mask = (face_indices == 1) & (ys < 0)
                u = (xs[mask] / abs_y[mask] + 1) / 2
                v = (zs[mask] / abs_y[mask] + 1) / 2
            elif face_name == 'front':
                mask = (face_indices == 2) & (zs > 0)
                u = (xs[mask] / abs_z[mask] + 1) / 2
                v = (ys[mask] / abs_z[mask] + 1) / 2
            elif face_name == 'back':
                mask = (face_indices == 2) & (zs < 0)
                u = (-xs[mask] / abs_z[mask] + 1) / 2
                v = (ys[mask] / abs_z[mask] + 1) / 2

            # Convert the face u, v coordinates to pixel coordinates
            face_height, face_width, _ = face_image.shape
            u_pixel = np.clip((u * face_width).astype(int), 0, face_width - 1)
            v_pixel = np.clip((v * face_height).astype(int),
                              0, face_height - 1)

            # Ensure mask is correctly shaped and boolean
            mask = mask.astype(bool)

            # Create boolean indices for equirectangular assignment
            masked_yv = yv[mask]
            masked_xv = xv[mask]

            # Ensure the index arrays are integer type
            masked_yv = masked_yv.astype(int)
            masked_xv = masked_xv.astype(int)

            # Get the color from the cubemap face and set it in the equirectangular image
            equirectangular_pixels[masked_yv,
                                   masked_xv] = face_image[v_pixel, u_pixel]

        # Convert the numpy array back to an image
        equirectangular_image = Image.fromarray(equirectangular_pixels)

        # Resize back to the desired output size
        if scale_factor > 1:
            equirectangular_image = equirectangular_image.resize(
                (output_width, output_height), Image.LANCZOS)

        self.logger.info('Converted Cube Map to Equirectangular Panorama.')

        return equirectangular_image
