import torch
import numpy as np
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
from diffusers.utils import load_image

def BEiTDepthEstimation(image: Image.Image, **kwargs) -> Image.Image:
    # Load model and processor once
    device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    processor = DPTImageProcessor.from_pretrained("Intel/dpt-beit-large-512")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-beit-large-512").to(device)
    model.eval()
    print("Model loaded on device:", device)

    image = load_image(image)

    # Prepare input
    inputs = processor(images=image, return_tensors="pt").to(device)
    print("Input prepared on device:", device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    print("Depth predicted on device:", device)
    # Interpolate while keeping tensors on the GPU
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    print("Depth interpolated on device:", device)

    # Move to CPU and convert to NumPy
    output = prediction.cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    print("Depth formatted on CPU")

    return Image.fromarray(formatted)