"""NSFW image classifier using SmilingWolf/wd-swinv2-tagger-v3 (ONNX).

Designed for AI-generated / anime content. Returns a raw float score (0.0-1.0)
representing NSFW confidence (max of 'questionable' and 'explicit' rating scores).
Threshold comparison is left to the caller.

Requires: onnxruntime (or onnxruntime-gpu), pandas, huggingface_hub
"""

import numpy as np
from PIL import Image

MODEL_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
IMAGE_SIZE = 448

_session = None
_tag_names = []
_rating_indices = {}   # {"general": idx, "sensitive": idx, "questionable": idx, "explicit": idx}


def _load_model() -> None:
    global _session, _tag_names, _rating_indices
    if _session is not None:
        return

    import onnxruntime as ort
    import pandas as pd
    from huggingface_hub import hf_hub_download

    model_path = hf_hub_download(MODEL_REPO, "model.onnx")
    csv_path   = hf_hub_download(MODEL_REPO, "selected_tags.csv")

    providers = ["CPUExecutionProvider"]
    _session = ort.InferenceSession(model_path, providers=providers)
    print(f"[NSFWClassifier] Provider: {_session.get_providers()[0]}")

    tags_df = pd.read_csv(csv_path)
    _tag_names = tags_df["name"].tolist()

    # Locate rating tags - support both "rating:xxx" and bare "xxx" naming
    rating_keywords = ("general", "sensitive", "questionable", "explicit")
    for idx, name in enumerate(tags_df["name"]):
        lower = name.lower()
        for kw in rating_keywords:
            if lower == kw or lower == f"rating:{kw}":
                _rating_indices[kw] = idx
                break

    found = list(_rating_indices.keys())
    missing = [kw for kw in rating_keywords if kw not in _rating_indices]
    print(f"[NSFWClassifier] Rating tags found {len(found)}/4: {found}")
    if missing:
        print(f"[NSFWClassifier] WARNING: missing rating tags: {missing}")


def _preprocess(image: Image.Image) -> np.ndarray:
    """Pad to square (white), resize to IMAGE_SIZE, float32 RGB NHWC (0-255)."""
    image = image.convert("RGB")
    w, h = image.size
    max_dim = max(w, h)
    canvas = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    canvas.paste(image, ((max_dim - w) // 2, (max_dim - h) // 2))
    canvas = canvas.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
    arr = np.array(canvas, dtype=np.float32)   # (H, W, C), RGB, 0-255
    return np.expand_dims(arr, 0)              # (1, H, W, C)


def score(image: Image.Image) -> float:
    """Return the raw NSFW confidence score (0.0-1.0).
    Combines 'questionable' and 'explicit' rating probabilities.
    Threshold comparison is left to the caller."""
    try:
        _load_model()
        if not _rating_indices:
            print("[NSFWClassifier] ERROR: no rating tags found, returning 0.0")
            return 0.0

        arr = _preprocess(image)
        input_name = _session.get_inputs()[0].name
        probs = _session.run(None, {input_name: arr})[0][0]

        q = float(probs[_rating_indices["questionable"]]) if "questionable" in _rating_indices else 0.0
        e = float(probs[_rating_indices["explicit"]])     if "explicit"     in _rating_indices else 0.0
        g = float(probs[_rating_indices["general"]])      if "general"      in _rating_indices else 0.0
        s = float(probs[_rating_indices["sensitive"]])    if "sensitive"    in _rating_indices else 0.0
        nsfw = max(q, e)
        print(f"[NSFWClassifier] g={g:.3f} s={s:.3f} q={q:.3f} e={e:.3f} -> nsfw_score={nsfw:.3f}")
        return nsfw
    except Exception as exc:
        import traceback
        print(f"[NSFWClassifier] Error: {exc}\n{traceback.format_exc()}")
        return 0.0
