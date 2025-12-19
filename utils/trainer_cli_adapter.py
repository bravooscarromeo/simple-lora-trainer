from pathlib import Path


def build_train_lora_cli_args(config: dict, project_dir: Path) -> list[str]:
    project = config["project"]
    model = config["model"]
    dataset = config["dataset"]
    training = config["training"]
    lora = config["lora"]
    precision = config["precision"]

    dataset_path = project_dir / dataset["path"]
    captions = dataset.get("captions", {})
    caption_ext = captions.get("extension", ".txt")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    if "architecture" not in model:
        raise ValueError("config.model.architecture is required")

    arch = model["architecture"]

    if arch in ("sd15", "sd1", "sd1.5", "sd"):
        model_type = "sd"
    elif arch == "sdxl":
        model_type = "sdxl"
    else:
        raise ValueError(
            f"Invalid model architecture '{arch}'. "
            "Expected one of: sd, sd15, sd1.5, sdxl"
        )

    checkpoint = model.get("checkpoint")
    if checkpoint:
        base_model = checkpoint
    else:
        base_model = (
            "stabilityai/stable-diffusion-xl-base-1.0"
            if model_type == "sdxl"
            else "runwayml/stable-diffusion-v1-5"
        )

    output_path = project_dir / "output" / f"{project['name']}.safetensors"

    args = [
        "--model_type", model_type,
        "--base_model", base_model,
        "--dataset", str(dataset_path),
        "--caption_ext", caption_ext,
        "--resolution", str(dataset["resolution"]),
        "--batch_size", str(dataset["batch_size"]),
        "--epochs", str(training["epochs"]),
        "--lora_rank", str(lora["rank"]),
        "--lora_alpha", str(lora["alpha"]),
        "--unet_lr", str(training["learning_rates"]["unet"]),
        "--clip_lr", str(training["learning_rates"]["clip"]),
        "--precision", precision["mixed_precision"],
        "--output", str(output_path),
        "--save_every_epochs", str(training.get("save_every_epochs", 0)),
    ]

    scheduler = config.get("scheduler", {})
    if "type" in scheduler:
        args += ["--scheduler_type", str(scheduler.get("type", "constant"))]
    else:
        args += ["--scheduler_type", str(training.get("scheduler_type", "constant"))]

    args += ["--warmup_steps", str(scheduler.get("warmup_steps", 0))]
    args += ["--num_cycles", str(scheduler.get("num_cycles", 1))]

    opt = config.get("optimizer", {})
    opt_type = str(opt.get("type", "adamw")).lower()
    args += ["--optimizer", opt_type]

    if opt_type == "sgd":
        args += ["--momentum", str(opt.get("momentum", 0.9))]
        if opt.get("nesterov", False):
            args.append("--nesterov")

    args += ["--weight_decay", str(opt.get("weight_decay", 0.01))]

    betas = opt.get("betas", [0.9, 0.999])
    if isinstance(betas, (list, tuple)) and len(betas) >= 2:
        args += ["--beta1", str(betas[0]), "--beta2", str(betas[1])]
    else:
        args += ["--beta1", "0.9", "--beta2", "0.999"]

    args += ["--epsilon", str(opt.get("epsilon", 1e-8))]

    args += ["--lora_dropout", str(lora.get("dropout", 0.0))]

    bucket = dataset.get("bucket", {})
    if bucket.get("enabled", False):
        args.append("--bucket")
        args += [
            "--bucket_min_res", str(bucket.get("min_res", 512)),
            "--bucket_max_res", str(bucket.get("max_res", 1536)),
            "--bucket_step", str(bucket.get("step", 64)),
        ]

    if precision.get("gradient_checkpointing", False):
        args.append("--gradient_checkpointing")

    args += ["--repeats", str(dataset.get("repeats", 1))]

    prepend = captions.get("prepend_token")
    append = captions.get("append_token")
    if prepend:
        args += ["--prepend_token", prepend]
    if append:
        args += ["--append_token", append]

    if captions.get("first_word_memorize", False):
        args.append("--memorize_first_token")

    if dataset.get("shuffle", False):
        args.append("--shuffle")

    if dataset.get("cache_latents", False):
        args.append("--cache_latents")

    ga = training.get("gradient_accumulation", 1)
    args += ["--grad_accum_steps", str(int(ga))]

    if training.get("do_inference", False):
        args.append("--do_inference")
        args += [
            "--inference_prompt", training.get("inference_prompt", "portrait photo, high quality"),
            "--inference_steps", str(training.get("inference_steps", 20)),
            "--inference_images", str(training.get("inference_images", 2)),
        ]

    conditioning = training.get("conditioning", {})
    clip_skip = conditioning.get("clip_skip", 0)
    args += ["--clip_skip", str(int(clip_skip))]

    targets = lora.get("target_modules")

    if isinstance(targets, str):
        if targets.lower() == "auto":
            pass
        else:
            parts = [t.strip() for t in targets.split(",") if t.strip()]
            if parts:
                args += ["--target_modules", ",".join(parts)]

    elif isinstance(targets, list) and targets:
        args += ["--target_modules", ",".join(targets)]

    if precision.get("xformers"):
        args.append("--use_xformers")

    if precision.get("cpu_offload"):
        args.append("--cpu_offload")

    return args
