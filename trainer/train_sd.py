import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.hf_cache import setup_hf_env
setup_hf_env()

import torch
from trainer.train.config import build_arg_parser, cfg_from_args, log, resolve_dtype, log_train_config
from trainer.train.data import build_dataset_buckets_and_tags, build_latent_cache
from trainer.train.meta import build_lora_metadata
from trainer.train.lora import DEFAULT_TARGET_MODULES, DEFAULT_TE_TARGET_MODULES, inject_lora, set_lora_scale, save_lora
from trainer.train.optim import build_optimizer, build_scheduler
from trainer.train.loop import train_epochs
from trainer.train.sd.models import load_sd_models
from trainer.train.sd.step import SDTrainStep
from trainer.train.sd.inference import run_inference_preview_in_memory
from trainer.train.time import ETATimer
from utils.ensure_models import ensure_base_model_available

def train(cfg):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype(cfg.precision)

    log(f"STATUS device={device.type} dtype={dtype}")
    log_train_config(cfg)
    ensure_base_model_available(cfg.base_model)
    
    if cfg.cpu_offload and not cfg.cache_latents:
        raise RuntimeError(
            "cpu_offload=True currently requires cache_latents=True "
            "(this trainer offloads VAE, so VAE must not be used in-step)."
        )

    if cfg.model_type != "sd":
        raise RuntimeError("train_lora_v1.py currently supports only --model_type sd (SD 1.x)")

    dataset, bucket_map, tag_counter, trained_words = build_dataset_buckets_and_tags(cfg)
    steps_per_epoch = sum(
        (len(ids) + cfg.batch_size - 1) // cfg.batch_size
        for ids in bucket_map.values()
    )
    updates_per_epoch = (steps_per_epoch + cfg.grad_accum_steps - 1) // cfg.grad_accum_steps
    total_opt_steps = updates_per_epoch * cfg.epochs

    timer = ETATimer(total_steps=total_opt_steps)

    log("STATUS loading_models")
    tokenizer, text_encoder, vae, unet, scheduler = load_sd_models(cfg, device, dtype)

    if cfg.use_xformers:
        try:
            unet.enable_xformers_memory_efficient_attention()
            log("STATUS xformers=ENABLED")
        except Exception as e:
            raise RuntimeError(
                "xFormers requested but could not be enabled. "
                "Is xformers installed?"
            ) from e
    else:
        log("STATUS xformers=DISABLED")

    train_clip = (cfg.clip_lr is not None) and (float(cfg.clip_lr) > 0.0)
    log(f"STATUS clip_train={train_clip} clip_lr={cfg.clip_lr}")

    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    text_encoder.eval()
    for p in text_encoder.parameters():
        p.requires_grad_(False)

    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    unet_targets = cfg.target_modules or DEFAULT_TARGET_MODULES
    unet_injected, unet_lora_params = inject_lora(unet, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout, unet_targets)

    assert next(unet.parameters()).is_cuda, "UNet must be fully on GPU during LoRA training"

    log(f"STATUS lora_targets={','.join(unet_targets)} matched={unet_injected}")
    log(f"STATUS lora_layers={unet_injected}")

    set_lora_scale(unet, 1.0)

    te_lora_params = []
    if train_clip:
        te_targets = DEFAULT_TE_TARGET_MODULES
        te_injected, te_lora_params = inject_lora(text_encoder, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout, te_targets)
        log(f"STATUS te_lora_targets={','.join(te_targets)} matched={te_injected}")

    param_groups = [{"params": unet_lora_params, "lr": cfg.unet_lr}]
    if train_clip:
        param_groups.append({"params": te_lora_params, "lr": cfg.clip_lr})

    trainable_params = list(unet_lora_params) + (list(te_lora_params) if train_clip else [])

    optimizer = build_optimizer(param_groups, cfg)
    lr_scheduler, _ = build_scheduler(cfg, optimizer, len(dataset))

    output_path = Path(cfg.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = output_path.stem

    cached_latents_by_bucket = build_latent_cache(
        cfg=cfg,
        dataset=dataset,
        bucket_map=bucket_map,
        vae=vae,
        device=device,
        dtype=dtype,
        scaling_factor=0.18215,
    )

    if cfg.cpu_offload:
        vae.to("cpu")
        log("STATUS cpu_offload=ENABLED components=vae")

        if not train_clip:
            text_encoder.to("cpu")
            log("STATUS cpu_offload=ENABLED components+=text_encoder")
        else:
            log("STATUS cpu_offload=PARTIAL text_encoder=GPU (train_clip=True)")
    else:
        log("STATUS cpu_offload=DISABLED")

    if cfg.do_inference and cfg.cpu_offload:
        log("WARN do_inference disabled because cpu_offload=True (preview expects GPU models)")
        cfg.do_inference = False

    step = SDTrainStep(
        cfg=cfg,
        dataset=dataset,
        cached_latents_by_bucket=cached_latents_by_bucket,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        scheduler=scheduler,
        device=device,
        dtype=dtype,
    )

    def on_epoch_end(epoch, state):
        if cfg.save_every_epochs > 0 and epoch % cfg.save_every_epochs == 0:
            out = output_dir / f"{base_name}_epoch_{epoch}.safetensors"
            log(f"STATUS saving checkpoint: {out.name}")
            metadata = build_lora_metadata(cfg, tag_counter, trained_words)
            save_lora(unet=unet, text_encoder=text_encoder if train_clip else None, path=str(out), metadata=metadata)

            if cfg.do_inference:
                preview_dir = output_dir / f"{base_name}_epoch_{epoch}_preview"
                preview_dir.mkdir(parents=True, exist_ok=True)

                prompt = cfg.inference_prompt

                log("STATUS inference preview: BASE (LoRA OFF)")
                set_lora_scale(unet, 0.0)
                run_inference_preview_in_memory(
                    unet=unet, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler,
                    output_dir=preview_dir / "base", prompt=prompt, steps=cfg.inference_steps,
                    num_images=cfg.inference_images, seed=cfg.seed, device=device, dtype=dtype, clip_skip=cfg.clip_skip
                )

                log("STATUS inference preview: LORA (LoRA ON)")
                set_lora_scale(unet, 1.0)
                run_inference_preview_in_memory(
                    unet=unet, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler,
                    output_dir=preview_dir / "lora", prompt=prompt, steps=cfg.inference_steps,
                    num_images=cfg.inference_images, seed=cfg.seed, device=device, dtype=dtype, clip_skip=cfg.clip_skip
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
    save_lora(unet=unet, text_encoder=text_encoder if train_clip else None, path=str(final_out), metadata=metadata)

def main():
    ap = build_arg_parser(default_resolution=512)
    args = ap.parse_args()
    cfg = cfg_from_args(args)
    train(cfg)

if __name__ == "__main__":
    main()
