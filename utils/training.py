from utils.ensure_fields import parse_int, parse_float


def apply(form, config, issues):
    """
    Apply training-related settings from form to config.
    Mutates config in-place.
    Appends validation issues.
    """

    training = config["training"]

    epochs = parse_int(form, "epochs", issues, min_value=1)
    if epochs is not None:
        training["epochs"] = epochs

    save_every = parse_int(form, "save_every_epochs", issues, min_value=1)
    if save_every is not None:
        training["save_every_epochs"] = save_every

    training["do_inference"] = "do_inference" in form

    grad_accum = parse_int(form, "gradient_accumulation", issues, min_value=1)
    if grad_accum is not None:
        training["gradient_accumulation"] = grad_accum

    conditioning = training["conditioning"]

    clip_skip = parse_int(form, "clip_skip", issues, min_value=0)
    if clip_skip is not None:
        conditioning["clip_skip"] = clip_skip

    lrs = training["learning_rates"]

    lr_unet = parse_float(form, "lr_unet", issues, min_value=0)
    if lr_unet is not None:
        lrs["unet"] = lr_unet

    lr_clip = parse_float(form, "lr_clip", issues, min_value=0)
    if lr_clip is not None:
        lrs["clip"] = lr_clip
