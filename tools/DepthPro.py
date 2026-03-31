import torch
import numpy as np
from PIL import Image
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

# Module-level cache
_processor = None
_model = None
_model_device = None

def DepthPro(image: Image.Image, colored: bool = False, **kwargs) -> Image.Image:
    global _processor, _model, _model_device
    
    if image.mode != "RGB":
        image = image.convert("RGB")

    device_str = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    if _model is None or _model_device != device:
        _processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
        _model = DepthProForDepthEstimation.from_pretrained(
            "apple/DepthPro-hf", 
            dtype=dtype, # Load in half-precision
            device_map=device  # Efficient loading
        )
        _model.eval()
        _model_device = device

    # 1. Faster preprocessing: Ensure image is RGB and reasonably sized
    original_w, original_h = image.size
    inputs = _processor(images=image, return_tensors="pt").to(device, dtype=dtype)

    # 2. Use inference_mode (faster than no_grad)
    with torch.inference_mode():
        outputs = _model(**inputs)
        predicted_depth = outputs.predicted_depth 

    # 3. GPU-side Post-processing (Avoid moving to CPU until the very end)
    # Ensure shape is (1, 1, H, W)
    if predicted_depth.dim() == 2:
        predicted_depth = predicted_depth.unsqueeze(0).unsqueeze(0)
    elif predicted_depth.dim() == 3:
        predicted_depth = predicted_depth.unsqueeze(1)

    # Normalize on GPU
    d_min = predicted_depth.min()
    d_max = predicted_depth.max()
    # Normalize to 0-1 (0 is near, 1 is far)
    normalized = (predicted_depth - d_min) / (d_max - d_min + 1e-8)

    if not colored:
        # ControlNet Grayscale logic: Near is White (1.0), Far is Black (0.0)
        # Invert and convert to uint8 on GPU
        grayscale = ((1.0 - normalized) * 255).clamp(0, 255).to(torch.uint8)
        # Move to CPU only once
        output_np = grayscale.squeeze().cpu().numpy()
        return Image.fromarray(output_np, mode="L").convert("RGB")
    else:
        # For colored heatmap, we move to CPU as Matplotlib needs it
        # Optimization: Don't import matplotlib unless strictly needed
        import matplotlib.cm as cm
        normalized_cpu = normalized.squeeze().cpu().float().numpy()
        colormap = cm.get_cmap("Spectral")
        rgb = colormap(normalized_cpu)[:, :, :3]
        return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")