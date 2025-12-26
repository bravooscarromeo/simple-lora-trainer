def apply(form, config, issues):
    model = config.setdefault("model", {})

    arch = form.get("model_architecture", "").strip()
    checkpoint = form.get("model_checkpoint", "").strip() or None

    if arch not in ("sdxl", "sd15"):
        issues.append({
            "field": "model_architecture",
            "level": "fatal",
            "message": "Unsupported model architecture."
        })
    else:
        model["architecture"] = arch

    model["checkpoint"] = checkpoint
