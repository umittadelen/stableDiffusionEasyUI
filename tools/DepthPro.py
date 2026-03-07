import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

# Module-level cache so the model loads only once per session
_processor = None
_model = None
_model_device = None


def DepthPro(image: Image.Image, colored: bool = False, **kwargs) -> Image.Image:
    """
    Run Apple DepthPro depth estimation.

    colored=False (default): returns a grayscale RGB image with near=white,
        far=black, suitable for feeding into a ControlNet depth pipeline.
    colored=True: returns a Spectral heatmap matching DepthPro's native look
        (back=blue, mid=yellow-orange, front=red), suitable for display.
    """
    global _processor, _model, _model_device

    device_str = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    if _model is None or _model_device != device:
        _processor = DepthProImageProcessorFast.from_pretrained("apple/DepthPro-hf")
        _model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(device)
        _model.eval()
        _model_device = device

    original_w, original_h = image.size
    inputs = _processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = _model(**inputs)
        predicted_depth = outputs.predicted_depth  # (H, W) or (1, H, W)

    # Ensure shape is (1, 1, H, W) for interpolate
    while predicted_depth.dim() < 4:
        predicted_depth = predicted_depth.unsqueeze(0)

    prediction = torch.nn.functional.interpolate(
        predicted_depth,
        size=(original_h, original_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()  # (H, W), metric depth: small=near, large=far

    # Normalize to [0, 1]: 0 = near, 1 = far
    d_min, d_max = prediction.min(), prediction.max()
    normalized = (prediction - d_min) / (d_max - d_min + 1e-8)

    if colored:
        # Spectral colormap applied directly:
        #   0 (near) → red/warm end
        #   1 (far)  → blue/cool end
        # This produces the classic DepthPro look: back=blue, front=red.
        colormap = cm.get_cmap("Spectral")
        rgb = colormap(normalized)[:, :, :3]  # drop alpha
        return Image.fromarray((rgb * 255).astype(np.uint8), mode="RGB")
    else:
        # Grayscale for ControlNet: near=white (255), far=black (0)
        # Invert normalized so near→1→255
        grayscale = ((1.0 - normalized) * 255).astype(np.uint8)
        return Image.fromarray(grayscale).convert("RGB")
