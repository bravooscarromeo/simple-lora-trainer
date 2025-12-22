from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

_processor: BlipProcessor | None = None
_model: BlipForConditionalGeneration | None = None
_device: str | None = None

def load(device: str = "cpu"):
    global _processor, _model, _device

    if _model is not None:
        return

    _device = device

    _processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )

    _model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    )
    _model.to(device)
    _model.eval()

    print(f"[BLIP] loaded on {device}")

import re

def _sanitize_caption(text: str) -> str:
    """
    Light cleanup to prevent BLIP repetition explosions.
    Keeps captions natural.
    """
    if not text:
        return ""

    text = text.strip()

    text = re.sub(
        r"\b(\w+)(?:\s+\1\b){2,}",
        r"\1 \1",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"(,\s*\w+)(?:\1){2,}",
        r"\1\1",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"^[,\s]+", "", text)

    if len(text) > 300:
        text = text[:300].rsplit(" ", 1)[0]

    return text.strip()

def generate_caption(image_path) -> str:
    if _model is None or _processor is None:
        raise RuntimeError("BLIP not loaded. Call load() first.")

    image = Image.open(image_path).convert("RGB")
    inputs = _processor(image, return_tensors="pt")

    if _device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=40
        )

    caption = _processor.decode(
        output[0],
        skip_special_tokens=True
    )
    
    caption = caption.strip()
    caption = _sanitize_caption(caption)
    return caption



def unload():
    global _processor, _model, _device

    if _model is not None:
        del _model
        _model = None

    if _processor is not None:
        del _processor
        _processor = None

    _device = None

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("[BLIP] unloaded")
