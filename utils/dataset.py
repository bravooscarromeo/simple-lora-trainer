from utils.ensure_fields import parse_int
from pathlib import Path
import re

def apply(form, config, issues):
    """
    Apply dataset-related settings from form to config.
    Mutates config in-place.
    """

    dataset = config["dataset"]

    if "dataset_path" in form:
        dataset["path"] = form["dataset_path"].strip()

    resolution = parse_int(form, "resolution", issues, min_value=64)
    if resolution is not None:
        dataset["resolution"] = resolution

    batch = parse_int(form, "batch_size", issues, min_value=1)
    if batch is not None:
        dataset["batch_size"] = batch

    repeats = parse_int(form, "repeats", issues, min_value=1)
    if repeats is not None:
        dataset["repeats"] = repeats

    dataset["shuffle"] = "shuffle" in form
    dataset["cache_latents"] = "cache_latents" in form

    captions = dataset["captions"]

    if "caption_extension" in form:
        captions["extension"] = form["caption_extension"].strip()

    captions["first_word_memorize"] = "first_word_memorize" in form

    if "prepend_token" in form:
        val = form["prepend_token"].strip()
        captions["prepend_token"] = val if val else None

    if "append_token" in form:
        val = form["append_token"].strip()
        captions["append_token"] = val if val else None

    bucket = dataset["bucket"]
    bucket_enabled = "bucket_enabled" in form
    bucket["enabled"] = bucket_enabled

    if bucket_enabled:
        min_res = parse_int(form, "bucket_min_res", issues, min_value=64)
        max_res = parse_int(form, "bucket_max_res", issues, min_value=64)
        step = parse_int(form, "bucket_step", issues, min_value=1)

        if min_res is not None:
            bucket["min_res"] = min_res
        if max_res is not None:
            bucket["max_res"] = max_res
        if step is not None:
            bucket["step"] = step

    resolution = dataset.get("resolution")
    model_arch = config.get("model", {}).get("architecture", "sdxl")

    if model_arch == "sd15" and resolution and resolution > 768:
        issues.append({
            "field": "resolution",
            "level": "fatal",
            "message": (
                "Resolution is too high for SD 1.5.\n"
                "SD 1.5 supports up to 768 resolution.\n\n"
                "Fix: lower resolution or switch to SDXL."
            )
        })
