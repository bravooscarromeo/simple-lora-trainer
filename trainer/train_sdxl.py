import torch
from pathlib import Path

from train.config import build_arg_parser, cfg_from_args, log, resolve_dtype, log_train_config
from train.data import build_dataset_buckets_and_tags, build_latent_cache
from train.meta import build_lora_metadata
from train.lora import DEFAULT_TARGET_MODULES, DEFAULT_TE_TARGET_MODULES, inject_lora, set_lora_scale, save_lora_sdxl
from train.optim import build_optimizer, build_scheduler
from train.loop import train_epochs
from train.sdxl.models import load_sdxl_components, load_sdxl_scheduler
from train.sdxl.step import SDXLTrainStep, encode_prompt_sdxl
from train.sdxl.inference import run_sdxl_inference_preview
from train.time import ETATimer

def train(cfg):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(cfg.precision)

    log(f"STATUS device={device.type} dtype={dtype}")
    log_train_config(cfg)

    if cfg.cpu_offload and not cfg.cache_latents:
        raise RuntimeError(
            "cpu_offload requires cache_latents=True "
            "(VAE runs on CPU only when latents are cached)"
        )

    if cfg.model_type != "sdxl":
        raise RuntimeError("train_lora_sdxl_v1.py supports only --model_type sdxl")

    dataset, bucket_map, tag_counter, trained_words = build_dataset_buckets_and_tags(cfg)
    steps_per_epoch = sum(
        (len(ids) + cfg.batch_size - 1) // cfg.batch_size
        for ids in bucket_map.values()
    )

    updates_per_epoch = (steps_per_epoch + cfg.grad_accum_steps - 1) // cfg.grad_accum_steps
    total_opt_steps = updates_per_epoch * cfg.epochs
    timer = ETATimer(total_steps=total_opt_steps)

    unet, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2 = load_sdxl_components(cfg.base_model, device, dtype)

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        log("STATUS gradient_checkpointing=ENABLED")
    else:
        log("STATUS gradient_checkpointing=DISABLED")

    if cfg.use_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            log("STATUS xformers=ENABLED (unet only)")
        except Exception:
            log("STATUS xformers=FAILED (continuing)")
    else:
        log("STATUS xformers=DISABLED")

    train_clip = cfg.clip_lr is not None and float(cfg.clip_lr) > 0.0
    log(f"STATUS clip_train={train_clip} clip_lr={cfg.clip_lr}")

    scheduler = load_sdxl_scheduler(cfg.base_model)

    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    text_encoder_2.eval()
    for p in text_encoder_2.parameters():
        p.requires_grad_(False)

    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    unet_targets = cfg.target_modules or DEFAULT_TARGET_MODULES
    unet_injected, unet_lora_params = inject_lora(unet, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout, unet_targets)

    assert next(unet.parameters()).is_cuda, "UNet must be fully on GPU during SDXL LoRA training"

    log(f"STATUS lora_targets={','.join(unet_targets)} matched={unet_injected}")
    log(f"STATUS lora_layers={unet_injected}")
    set_lora_scale(unet, 1.0)

    te1_lora_params = []
    te2_lora_params = []
    if train_clip:
        te_targets = DEFAULT_TE_TARGET_MODULES
        te1_injected, te1_lora_params = inject_lora(text_encoder, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout, te_targets)
        te2_injected, te2_lora_params = inject_lora(text_encoder_2, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout, te_targets)
        log(f"STATUS te1_lora_targets={','.join(te_targets)} matched={te1_injected}")
        log(f"STATUS te2_lora_targets={','.join(te_targets)} matched={te2_injected}")

    param_groups = [{"params": unet_lora_params, "lr": cfg.unet_lr}]
    if train_clip:
        param_groups.append({"params": te1_lora_params, "lr": cfg.clip_lr})
        param_groups.append({"params": te2_lora_params, "lr": cfg.clip_lr})

    trainable_params = list(unet_lora_params) + (list(te1_lora_params) + list(te2_lora_params) if train_clip else [])

    optimizer = build_optimizer(param_groups, cfg)
    lr_scheduler, _ = build_scheduler(cfg, optimizer, len(dataset))

    output_path = Path(cfg.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = output_path.stem

    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))

    cached_latents_by_bucket = build_latent_cache(
        cfg=cfg,
        dataset=dataset,
        bucket_map=bucket_map,
        vae=vae,
        device=device,
        dtype=dtype,
        scaling_factor=scaling_factor,
    )

    if cfg.cpu_offload:
        vae.to("cpu")
        log("STATUS cpu_offload=ENABLED components=vae")

        if not train_clip:
            text_encoder.to("cpu")
            text_encoder_2.to("cpu")
            log("STATUS cpu_offload=ENABLED components+=text_encoder,text_encoder_2")
        else:
            log("STATUS cpu_offload=PARTIAL text_encoders=GPU (train_clip=True)")
    else:
        log("STATUS cpu_offload=DISABLED")

    if cfg.cpu_offload:
        assert next(unet.parameters()).is_cuda, "cpu_offload must NOT move UNet off GPU during training"

    if cfg.do_inference and cfg.cpu_offload:
        log("WARN do_inference disabled because cpu_offload=True (preview expects GPU models)")
        cfg.do_inference = False

    step = SDXLTrainStep(
        cfg=cfg,
        dataset=dataset,
        cached_latents_by_bucket=cached_latents_by_bucket,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        device=device,
        dtype=dtype,
        scaling_factor=scaling_factor,
    )

    def on_epoch_end(epoch, state):
        if cfg.save_every_epochs > 0 and epoch % cfg.save_every_epochs == 0:
            out = output_dir / f"{base_name}_epoch_{epoch}.safetensors"
            log(f"STATUS saving checkpoint: {out.name}")
            metadata = build_lora_metadata(cfg, tag_counter, trained_words)
            save_lora_sdxl(
                unet=unet,
                text_encoder=text_encoder if train_clip else None,
                text_encoder_2=text_encoder_2 if train_clip else None,
                path=str(out),
                metadata=metadata,
            )

        if cfg.do_inference:
            preview_dir = output_dir / f"{base_name}_epoch_{epoch}_preview"
            log("STATUS inference preview (SDXL)")
            prompt_embeds, pooled = encode_prompt_sdxl(
                [cfg.inference_prompt], tokenizer, tokenizer_2, text_encoder, text_encoder_2, dtype
            )
            run_sdxl_inference_preview(
                unet=unet,
                vae=vae,
                scheduler=scheduler,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled,
                output_dir=preview_dir,
                steps=cfg.inference_steps,
                seed=cfg.seed,
                dtype=dtype,
                resolution=cfg.resolution,
            )

    train_epochs(
        cfg=cfg,
        dataset=dataset,
        bucket_map=bucket_map,
        step_fn=step,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        trainable_params=trainable_params,
        on_epoch_end=on_epoch_end,
        timer=timer,
    )

    final_out = output_dir / f"{base_name}_final.safetensors"
    log(f"STATUS saving final: {final_out.name}")
    metadata = build_lora_metadata(cfg, tag_counter, trained_words)
    save_lora_sdxl(
        unet=unet,
        text_encoder=text_encoder if train_clip else None,
        text_encoder_2=text_encoder_2 if train_clip else None,
        path=str(final_out),
        metadata=metadata,
    )

def main():
    ap = build_arg_parser(default_resolution=1024)
    args = ap.parse_args()
    cfg = cfg_from_args(args)
    train(cfg)

if __name__ == "__main__":
    main()
