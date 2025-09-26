import numpy as np
from PIL import Image

# convert numpy to Image
def numpy_to_image(array: np.ndarray) -> Image.Image:
    """Convert a NumPy array to a PIL Image."""
    array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array)

# convert Image to numpy
def image_to_numpy(image: Image.Image) -> np.ndarray:
    """Convert a PIL Image to a NumPy array."""
    return np.array(image).astype(np.float32) / 255.0