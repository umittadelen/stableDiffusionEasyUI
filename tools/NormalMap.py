import torch
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation

def NormalMap(image: Image.Image, **kwargs) -> Image.Image:
    # Determine device
    device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)
    
    # Load depth estimation model
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to(device)
    
    # Preprocess image
    original_size = image.size
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Run depth estimation
    with torch.no_grad():
        depth = model(**inputs).predicted_depth
    
    # Convert depth to numpy array
    depth = depth.squeeze().cpu().numpy()
    
    # Resize depth to match original image size
    depth = np.array(Image.fromarray(depth).resize(original_size, Image.BILINEAR))
    
    # Compute normals
    dzdx = np.gradient(depth, axis=1)
    dzdy = np.gradient(depth, axis=0)
    
    normal = np.dstack((-dzdx, -dzdy, np.ones_like(depth)))
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal /= norm  # Normalize to unit vector
    
    # Convert to RGB format
    normal = (normal + 1) / 2  # Scale to [0, 1]
    normal = (normal * 255).astype(np.uint8)  # Scale to [0, 255]
    normal_image = Image.fromarray(normal, mode="RGB")
    
    return normal_image
