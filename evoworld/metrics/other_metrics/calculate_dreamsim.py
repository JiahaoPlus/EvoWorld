from dreamsim import dreamsim
from PIL import Image


def calculate_dreamsim(image1_path, image2_path):
    """Calculate the DReAMSim score between two images."""
    # Load images
    image1 = Image.open(image1_path).convert("RGB")
    image2 = Image.open(image2_path).convert("RGB")
    device = "cuda"
    model, preprocess = dreamsim(pretrained=True, device=device)
    
    img1 = preprocess(image1).to(device)
    img2 = preprocess(image2).to(device)
    # Calculate DReAMSim score
    score = model(image1, image2)
    return score
