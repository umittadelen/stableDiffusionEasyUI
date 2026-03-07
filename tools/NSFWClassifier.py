"""NSFW image classifier using Falconsai/nsfw_image_detection_26.

Returns 'nsfw' or 'normal'. Model is lazy-loaded on first call.
Requires: transformers, torch
"""

import torch
from PIL import Image

REPO_ID = "Falconsai/nsfw_image_detection_26"

_classifier = None


def _load_model() -> None:
    global _classifier
    if _classifier is not None:
        return
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    _classifier = pipeline("image-classification", model=REPO_ID, device=device)
    print(f"[NSFWClassifier] Model loaded on {'CUDA' if device == 0 else 'CPU'}")


def classify(image: Image.Image, threshold: float = 0.5) -> str:
    """Return 'nsfw' or 'normal'. threshold is the minimum confidence (0-1)
    required to classify an image as nsfw (default 0.5)."""
    try:
        _load_model()
        results = _classifier(image.convert("RGB"))
        nsfw_score = next((r["score"] for r in results if r["label"].lower() == "nsfw"), 0.0)
        rating = "nsfw" if nsfw_score >= threshold else "normal"
        print(f"[NSFWClassifier] nsfw_score={nsfw_score:.3f} threshold={threshold} -> {rating}")
        return rating
    except Exception as exc:
        import traceback
        print(f"[NSFWClassifier] Error: {exc}\n{traceback.format_exc()}")
        return "normal"
