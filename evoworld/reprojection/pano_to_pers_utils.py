
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
UNITY_TO_OPENCV = [1, -1, 1, -1, 1, -1]
def calculate_segment_indices(segment_id: int) -> Tuple[int, int]:
    """Calculate start and end indices for the segment."""
    look_at_idx = (segment_id + 1) * 24 + 24
    start_idx = segment_id * 24 + 1

    if segment_id == 0:
        start_idx = start_idx - 1

    end_idx = start_idx + 25
    return start_idx, end_idx, look_at_idx


def read_camera_file_and_convert_to_rdf(camera_file: str) -> np.ndarray:
    """Read camera parameters file and convert to RDF format."""
    with open(camera_file, 'r') as f:
        lines = f.readlines()

    # Convert each line to list of floats, skip first line (header)
    camera_params = [list(map(float, line.strip().split(',')))[1:] for line in lines[1:]]
    camera_params = np.array(camera_params)

    # Convert to RDF
    camera_params = camera_params * UNITY_TO_OPENCV
    return camera_params

def write_camera_file(camera_params: np.ndarray, output_camera_file: str):
    """Write camera parameters to file."""
    with open(output_camera_file, 'w') as f:
        for i in range(len(camera_params)):
            f.write(f"{i+1} {camera_params[i][0]} {camera_params[i][1]} {camera_params[i][2]} "
                   f"{camera_params[i][3]} {camera_params[i][4]} {camera_params[i][5]}\n")