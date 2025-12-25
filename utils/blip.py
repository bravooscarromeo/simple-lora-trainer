from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import re

_processor: BlipProcessor | None = None
_model: BlipForConditionalGeneration | None = None
_device: str | None = None

FILLER_PHRASES = ("a photo of", "a picture of", "there is", "there are", "this is")
TAIL_KEYWORDS = ("lighting", "light", "shadow", "depth of field", "bokeh", "cinematic", "dramatic", "soft", "natural lighting", "studio lighting")
OBJECT_PREFIXES = ("a piece of", "a cup of", "a glass of", "a bottle of", "a bowl of", "a plate of")
SUBJECT_WORDS = ("man", "woman", "person", "girl", "boy", "people")
POSE_WORDS = ("standing", "sitting", "walking", "lying", "kneeling")
SETTING_WORDS = ("outdoor", "indoors", "room", "street", "studio", "nature")
MEDIUM_WORDS = ("photo", "photograph", "illustration", "drawing", "painting")
POSE_VERBS = ("posing", "standing", "sitting", "walking", "lying")

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

def _dedupe_tags(tags: list[str]) -> list[str]:
    seen = set()
    out = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def generate_caption(image_path, add_tail: bool = False) -> str:
    if _model is None or _processor is None:
        raise RuntimeError("BLIP not loaded. Call load() first.")

    image = Image.open(image_path).convert("RGB")
    inputs = _processor(image, return_tensors="pt")

    if _device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=32,
            num_beams=5,
            do_sample=False,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )

    caption = _processor.decode(
        output[0],
        skip_special_tokens=True
    )

    caption = sentence_to_tags(caption, add_tail=add_tail)

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

def _collapse_repeated_words(tag: str) -> str:
    words = tag.split()
    out = []
    for w in words:
        if not out or out[-1] != w:
            out.append(w)
    return " ".join(out)

def _explode_tag(tag: str) -> list[str]:
    parts = tag.split()
    if len(parts) > 2 and parts[-1] in POSE_VERBS:
        return [" ".join(parts[:-1]), parts[-1]]
    return [tag]

def _strip_weird_chars(text: str) -> str:
    return re.sub(r"[^a-z0-9 ,\.]", " ", text)

def _strip_copula(text: str) -> str:
    return re.sub(
        r"\b(is|are|was|were|be|being)\s+(\w+)",
        r"\2",
        text,
        flags=re.IGNORECASE,
    )

def _normalize_object(text: str) -> str:
    for p in OBJECT_PREFIXES:
        if text.startswith(p):
            return text.replace(p, "", 1).strip()
    return text

def _normalize_sentence(text: str) -> str:
    text = text.lower()
    for f in FILLER_PHRASES:
        text = text.replace(f, "")
    text = _strip_copula(text)
    text = _strip_weird_chars(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _split_chunks(text: str) -> list[str]:
    return re.split(
        r",| and | with | wearing | holding | standing | sitting | in | on ",
        text,
    )

def _clean_tag(tag: str) -> str:
    tag = tag.strip().lower()
    tag = re.sub(r"^(a |an |the )", "", tag)
    tag = _normalize_object(tag)
    tag = tag.replace(".", "").replace(",", "")
    tag = _collapse_repeated_words(tag)
    return tag.strip()

def _is_tail(chunk: str) -> bool:
    return any(k in chunk for k in TAIL_KEYWORDS)

def _order_tags(tags: list[str]) -> list[str]:
    ordered = []
    remaining = tags[:]

    def pull(predicate):
        nonlocal remaining
        pulled = [t for t in remaining if predicate(t)]
        remaining = [t for t in remaining if t not in pulled]
        return pulled

    ordered += pull(lambda t: any(w in t for w in SUBJECT_WORDS))
    ordered += pull(lambda t: "hair" in t or "face" in t)
    ordered += pull(lambda t: any(w in t for w in ("shirt", "pants", "jeans", "dress")))
    ordered += pull(lambda t: any(w in t for w in POSE_WORDS))
    ordered += pull(lambda t: any(w in t for w in SETTING_WORDS))
    ordered += pull(lambda t: any(w in t for w in MEDIUM_WORDS))
    ordered += remaining

    return ordered

def sentence_to_tags(
    sentence: str,
    add_tail: bool = False,
) -> str:
    sentence = _normalize_sentence(sentence)
    chunks = _split_chunks(sentence)

    tags: list[str] = []
    tail: list[str] = []

    for c in chunks:
        c = c.strip()
        if not c:
            continue

        if _is_tail(c):
            tail.append(c)
        else:
            cleaned = _clean_tag(c)
            for t in _explode_tag(cleaned):
                if t:
                    tags.append(t)

    tags = _dedupe_tags(tags)
    tags = _order_tags(tags)
    caption = ", ".join(tags)

    if add_tail and tail:
        caption += ", " + " ".join(tail[:1])

    return caption
