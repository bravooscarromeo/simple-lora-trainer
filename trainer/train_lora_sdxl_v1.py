import os
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import List, Tuple
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms

from diffusers import StableDiffusionXLPipeline, DDPMScheduler
from safetensors.torch import save_file
from diffusers.optimization import get_scheduler

@dataclass
class TrainConfig:
    model_type: str
    base_model: str
    dataset: str
    caption_ext: str
    resolution: int
    batch_size: int
    epochs: int
    shuffle: bool
    lora_rank: int
    lora_alpha: float
    unet_lr: float
    clip_lr: float
    precision: str
    output: str
    gradient_checkpointing: bool = False
    grad_accum_steps: int = 1
    repeats: int = 1
    save_every_epochs: int = 0
    seed: int = 0
    log_every: int = 10
    do_inference: bool = False
    inference_prompt: str = ""
    inference_steps: int = 20
    inference_images: int = 2
    clip_skip: int = 0
    scheduler_type: str = "constant"
    warmup_steps: int = 0
    num_cycles: int = 1
    prepend_token: str | None = None
    append_token: str | None = None
    cache_latents: bool = False
    bucket_enabled: bool = False
    bucket_min_res: int = 512
    bucket_max_res: int = 1536
    bucket_step: int = 64
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    memorize_first_token: bool = False
    lora_dropout: float = 0.0
    momentum: float = 0.9
    nesterov: bool = False
    target_modules: list[str] | None = None
    use_xformers: bool = False
    cpu_offload: bool = False

def parse_caption_tags(text: str) -> list[str]:
    parts = [t.strip() for t in text.split(",")]
    return [p for p in parts if p]


def build_lora_metadata(cfg: TrainConfig, tag_counter: Counter, trained_words: str) -> dict[str, str]:
    tag_payload = {
        "dataset": dict(tag_counter)
    }

    return {
        "ss_tag_frequency": json.dumps(tag_payload, ensure_ascii=False),
        "ss_trained_words": trained_words,
        "ss_network_dim": str(cfg.lora_rank),
        "ss_network_alpha": str(cfg.lora_alpha),
    }

def log(msg: str) -> None:
    print(msg, flush=True)

def resolve_dtype(precision: str) -> torch.dtype:
    if precision in ("fp16", "float16"):
        return torch.float16
    if precision in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32

def log_train_config(cfg: TrainConfig) -> None:
    log("===== TRAIN CONFIG =====")
    log(f"model_type={cfg.model_type}")
    log(f"base_model={cfg.base_model}")
    log(f"precision={cfg.precision}")
    log("device=cuda")
    log(f"resolution={cfg.resolution}")
    log(f"batch_size={cfg.batch_size}")
    log(f"grad_accum_steps={cfg.grad_accum_steps}")
    log(f"effective_batch_size={cfg.batch_size * cfg.grad_accum_steps}")
    log(f"epochs={cfg.epochs}")
    log(f"repeats={cfg.repeats}")
    log(f"optimizer={cfg.optimizer}")
    log(f"unet_lr={cfg.unet_lr}")
    log(f"clip_lr={cfg.clip_lr}")
    log(f"weight_decay={cfg.weight_decay}")
    log(f"betas=({cfg.beta1}, {cfg.beta2})")
    log(f"epsilon={cfg.epsilon}")
    log(f"momentum={cfg.momentum}")
    log(f"nesterov={cfg.nesterov}")
    log(f"scheduler={cfg.scheduler_type}")
    log(f"warmup_steps={cfg.warmup_steps}")
    log(f"num_cycles={cfg.num_cycles}")
    log(f"lora_rank={cfg.lora_rank}")
    log(f"lora_alpha={cfg.lora_alpha}")
    log(f"lora_dropout={cfg.lora_dropout}")
    log(f"train_clip={cfg.clip_lr > 0}")
    log(f"cache_latents={cfg.cache_latents}")
    log(f"bucket_enabled={cfg.bucket_enabled}")
    if cfg.bucket_enabled:
        log(f"bucket_min={cfg.bucket_min_res} max={cfg.bucket_max_res} step={cfg.bucket_step}")
    log(f"prepend_token={cfg.prepend_token}")
    log(f"append_token={cfg.append_token}")
    log(f"memorize_first_token={cfg.memorize_first_token}")
    log(f"do_inference={cfg.do_inference} (ignored for now)")
    log("===== END CONFIG =====")

def load_dataset(dataset_dir: str, caption_ext: str) -> List[Tuple[str, str]]:
    items = []
    for name in sorted(os.listdir(dataset_dir)):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            img_path = os.path.join(dataset_dir, name)
            cap_path = os.path.join(dataset_dir, os.path.splitext(name)[0] + caption_ext)
            if not os.path.exists(cap_path):
                raise RuntimeError(f"Missing caption for {name}")
            items.append((img_path, cap_path))
    if not items:
        raise RuntimeError("Dataset is empty")
    return items

def image_transform(res: int):
    return transforms.Compose([
        transforms.Resize(res, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(res),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

def pick_bucket_resolution(w: int, h: int, cfg: TrainConfig) -> int:
    base = min(max(max(w, h), cfg.bucket_min_res), cfg.bucket_max_res)
    step = cfg.bucket_step
    return (base // step) * step

DEFAULT_TARGET_MODULES = ["to_q", "to_k", "to_v", "to_out.0"]
DEFAULT_TE_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "out_proj"]

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_scale = 1.0
        self.dropout = dropout

        device = base.weight.device
        dtype = base.weight.dtype

        self.A = nn.Parameter(torch.randn(base.in_features, rank, device=device, dtype=dtype) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, base.out_features, device=device, dtype=dtype))

    def forward(self, x, *args, **kwargs):
        delta = self.scale * ((x @ self.A) @ self.B)
        if self.training and self.dropout > 0:
            delta = F.dropout(delta, p=self.dropout)
        return self.base(x, *args, **kwargs) + self.lora_scale * delta

def parse_target_modules(s: str) -> list[str] | None:
    s = (s or "").strip()
    if not s:
        return None
    raw = [t.strip() for t in s.replace(" ", ",").split(",")]
    out = [t for t in raw if t]
    return out or None

def inject_lora(unet: nn.Module, rank: int, alpha: float, dropout: float, targets: list[str]) -> int:
    replaced = 0
    modules = dict(unet.named_modules())

    for name, mod in list(modules.items()):
        if isinstance(mod, nn.Linear) and any(name.endswith(suf) for suf in targets):
            parent_name, child = name.rsplit(".", 1)
            parent = modules[parent_name]
            setattr(parent, child, LoRALinear(mod, rank, alpha, dropout))
            replaced += 1

    if replaced == 0:
        raise RuntimeError("Injected 0 LoRA layers (SDXL module names may differ)")

    return replaced

def inject_lora_text_encoder(te: nn.Module, rank: int, alpha: float, dropout: float, targets: list[str]):
    replaced = 0
    modules = dict(te.named_modules())

    for name, mod in list(modules.items()):
        if isinstance(mod, nn.Linear) and any(name.endswith(suf) for suf in targets):
            parent_name, child = name.rsplit(".", 1)
            parent = modules[parent_name]
            setattr(parent, child, LoRALinear(mod, rank, alpha, dropout))
            replaced += 1

    if replaced == 0:
        raise RuntimeError("Injected 0 text encoder LoRA layers")

    params = list(lora_parameters(te))
    return replaced, params

def lora_parameters(unet: nn.Module):
    for m in unet.modules():
        if isinstance(m, LoRALinear):
            yield m.A
            yield m.B


def save_lora_sdxl(
    *,
    unet: nn.Module,
    text_encoder: nn.Module | None,
    text_encoder_2: nn.Module | None,
    path: str,
    metadata: dict | None = None,
):
    tensors = {}

    for name, m in unet.named_modules():
        if not isinstance(m, LoRALinear):
            continue

        key = "lora_unet_" + name.replace(".", "_")

        tensors[f"{key}.lora_down.weight"] = (
            m.A.T.detach().float().contiguous().cpu()
        )
        tensors[f"{key}.lora_up.weight"] = (
            m.B.T.detach().float().contiguous().cpu()
        )
        tensors[f"{key}.alpha"] = torch.tensor(m.alpha)

    if text_encoder is not None:
        for name, m in text_encoder.named_modules():
            if not isinstance(m, LoRALinear):
                continue

            key = "lora_te1_" + name.replace(".", "_")

            tensors[f"{key}.lora_down.weight"] = (
                m.A.T.detach().float().contiguous().cpu()
            )
            tensors[f"{key}.lora_up.weight"] = (
                m.B.T.detach().float().contiguous().cpu()
            )
            tensors[f"{key}.alpha"] = torch.tensor(m.alpha)

    if text_encoder_2 is not None:
        for name, m in text_encoder_2.named_modules():
            if not isinstance(m, LoRALinear):
                continue

            key = "lora_te2_" + name.replace(".", "_")

            tensors[f"{key}.lora_down.weight"] = (
                m.A.T.detach().float().contiguous().cpu()
            )
            tensors[f"{key}.lora_up.weight"] = (
                m.B.T.detach().float().contiguous().cpu()
            )
            tensors[f"{key}.alpha"] = torch.tensor(m.alpha)

    save_file(
        tensors,
        path,
        metadata=metadata or {}
    )

@torch.no_grad()
def encode_prompt_sdxl(
    pipe: StableDiffusionXLPipeline,
    prompts: list[str],
    device: torch.device,
    dtype: torch.dtype,
):
    out = pipe.encode_prompt(
        prompt=prompts,
        prompt_2=prompts,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=False,
    )

    if isinstance(out, tuple):
        if len(out) == 2:
            prompt_embeds, pooled = out
        elif len(out) == 3:
            prompt_embeds, pooled, _ = out
        elif len(out) >= 4:
            prompt_embeds = out[0]
            pooled = out[2]
        else:
            raise RuntimeError(f"Unexpected encode_prompt return length: {len(out)}")
    else:
        raise RuntimeError("encode_prompt did not return a tuple")

    return prompt_embeds.to(dtype), pooled.to(dtype)

def make_add_time_ids(batch_size: int, bucket_res: int, device: torch.device, dtype: torch.dtype):
    h = w = int(bucket_res)
    t = torch.tensor([h, w, 0, 0, h, w], device=device, dtype=dtype)
    return t.unsqueeze(0).repeat(batch_size, 1)

def load_sdxl_pipeline(cfg: TrainConfig, device: torch.device, dtype: torch.dtype) -> StableDiffusionXLPipeline:
    pipe = StableDiffusionXLPipeline.from_pretrained(cfg.base_model, torch_dtype=dtype)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if cfg.gradient_checkpointing:
        pipe.unet.enable_gradient_checkpointing()
        log("STATUS gradient_checkpointing=ENABLED")
    else:
        log("STATUS gradient_checkpointing=DISABLED")

    return pipe

def train(cfg: TrainConfig):
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

    log("STATUS loading_dataset")
    base_dataset = load_dataset(cfg.dataset, cfg.caption_ext)

    if cfg.repeats < 1:
        raise ValueError("repeats must be >= 1")

    dataset = base_dataset * cfg.repeats

    tag_counter = Counter()

    for _, cap_path in dataset:
        text = open(cap_path, "r", encoding="utf-8").read().strip()

        if cfg.memorize_first_token:
            first = text.split(",")[0].strip()
            if first:
                text = f"{first}, {text}"

        if cfg.prepend_token:
            text = f"{cfg.prepend_token}, {text}"
        if cfg.append_token:
            text = f"{text}, {cfg.append_token}"

        tag_counter.update(parse_caption_tags(text))

    trained_words = ", ".join([t for t, _ in tag_counter.most_common(200)])

    bucket_map: dict[int, list[int]] = {}

    if cfg.bucket_enabled:
        for i, (img_path, _) in enumerate(dataset):
            with Image.open(img_path) as img:
                w, h = img.size
            bucket_res = pick_bucket_resolution(w, h, cfg)
            bucket_map.setdefault(bucket_res, []).append(i)
    else:
        bucket_map[cfg.resolution] = list(range(len(dataset)))

    for res, ids in bucket_map.items():
        log(f"STATUS bucket[{res}] size={len(ids)}")

    if cfg.cache_latents and cfg.bucket_enabled:
        log("STATUS cache_latents=ENABLED (per-bucket)")
    elif cfg.cache_latents:
        log("STATUS cache_latents=ENABLED")

    if cfg.bucket_enabled:
        log(
            f"STATUS bucket=ENABLED min={cfg.bucket_min_res} max={cfg.bucket_max_res} step={cfg.bucket_step}"
        )
    else:
        log("STATUS bucket=DISABLED")

    if cfg.model_type != "sdxl":
        raise RuntimeError("train_lora_sdxl_v1.py supports only --model_type sdxl")

    log("STATUS loading_models")
    pipe = load_sdxl_pipeline(cfg, device, dtype)

    if cfg.use_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            log("STATUS xformers=ENABLED")
        except Exception as e:
            raise RuntimeError(
                "xFormers requested but could not be enabled. "
                "Is xformers installed?"
            ) from e
    else:
        log("STATUS xformers=DISABLED")

    unet = pipe.unet
    vae = pipe.vae

    text_encoder = pipe.text_encoder
    text_encoder_2 = pipe.text_encoder_2
    train_clip = cfg.clip_lr is not None and float(cfg.clip_lr) > 0.0
    log(f"STATUS clip_train={train_clip} clip_lr={cfg.clip_lr}")

    scheduler = DDPMScheduler.from_pretrained(cfg.base_model, subfolder="scheduler")

    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    pipe.text_encoder.eval()
    for p in pipe.text_encoder.parameters():
        p.requires_grad_(False)

    pipe.text_encoder_2.eval()
    for p in pipe.text_encoder_2.parameters():
        p.requires_grad_(False)

    unet.eval()
    for p in unet.parameters():
        p.requires_grad_(False)

    unet_targets = cfg.target_modules or DEFAULT_TARGET_MODULES
    unet_injected = inject_lora(unet, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout, unet_targets)
    unet_lora_params = list(lora_parameters(unet))

    log(f"STATUS lora_targets={','.join(unet_targets)} matched={unet_injected}")
    log(f"STATUS lora_layers={unet_injected}")

    for m in unet.modules():
        if isinstance(m, LoRALinear):
            m.lora_scale = 1.0

    te1_lora_params = []
    te2_lora_params = []

    if train_clip:
        te_targets = DEFAULT_TE_TARGET_MODULES

        te1_injected, te1_lora_params = inject_lora_text_encoder(
            text_encoder, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout, te_targets
        )
        te2_injected, te2_lora_params = inject_lora_text_encoder(
            text_encoder_2, cfg.lora_rank, cfg.lora_alpha, cfg.lora_dropout, te_targets
        )

        log(f"STATUS te1_lora_targets={','.join(te_targets)} matched={te1_injected}")
        log(f"STATUS te2_lora_targets={','.join(te_targets)} matched={te2_injected}")

    param_groups = [{"params": unet_lora_params, "lr": cfg.unet_lr}]
    if train_clip:
        param_groups.append({"params": te1_lora_params, "lr": cfg.clip_lr})
        param_groups.append({"params": te2_lora_params, "lr": cfg.clip_lr})

    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.epsilon,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            betas=(cfg.beta1, cfg.beta2),
            eps=cfg.epsilon,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            momentum=cfg.momentum,
            nesterov=cfg.nesterov,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    lrs = [pg["lr"] for pg in optimizer.param_groups]
    log(f"STATUS optimizer_param_group_lrs={lrs}")

    steps_per_epoch = (len(dataset) + cfg.batch_size - 1) // cfg.batch_size
    updates_per_epoch = (steps_per_epoch + cfg.grad_accum_steps - 1) // cfg.grad_accum_steps
    num_training_steps = updates_per_epoch * cfg.epochs

    lr_scheduler = get_scheduler(
        name=cfg.scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=cfg.num_cycles,
    )

    log(
        f"STATUS training_plan steps_per_epoch={steps_per_epoch} "
        f"updates_per_epoch={updates_per_epoch} total_updates={num_training_steps}"
    )

    if cfg.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")

    optimizer.zero_grad(set_to_none=True)
    opt_step = 0
    global_step = 0

    output_path = Path(cfg.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = output_path.stem

    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))

    cached_latents_by_bucket: dict[int, dict[int, torch.Tensor]] | None = None
    if cfg.cache_latents:
        log("STATUS building latent cache")
        cached_latents_by_bucket = {}

        with torch.no_grad():
            for bucket_res, bucket_indices in bucket_map.items():
                log(f"STATUS caching bucket_res={bucket_res} samples={len(bucket_indices)}")
                tfm_bucket = image_transform(bucket_res)
                bucket_cache: dict[int, torch.Tensor] = {}

                for idx in bucket_indices:
                    img_path, _ = dataset[idx]
                    img = Image.open(img_path).convert("RGB")
                    pixel = tfm_bucket(img).unsqueeze(0).to(device=device, dtype=dtype)

                    latents = vae.encode(pixel).latent_dist.sample() * scaling_factor
                    bucket_cache[idx] = latents.detach().to(torch.float16).cpu()

                cached_latents_by_bucket[bucket_res] = bucket_cache

        total_cached = sum(len(v) for v in cached_latents_by_bucket.values())
        log(f"STATUS cached_latents_total={total_cached}")

    if cfg.cpu_offload:
        pipe.vae.to("cpu")
        log("STATUS cpu_offload=ENABLED components=vae")

        if not train_clip:
            pipe.text_encoder.to("cpu")
            pipe.text_encoder_2.to("cpu")
            log("STATUS cpu_offload=ENABLED components+=text_encoder,text_encoder_2")
        else:
            log("STATUS cpu_offload=PARTIAL text_encoders=GPU (train_clip=True)")
    else:
        log("STATUS cpu_offload=DISABLED")

    for epoch in range(1, cfg.epochs + 1):

        for bucket_res, bucket_indices in bucket_map.items():
            log(f"STATUS training bucket_res={bucket_res} samples={len(bucket_indices)}")

            tfm_bucket = image_transform(bucket_res)

            if cfg.shuffle:
                bucket_indices = torch.tensor(bucket_indices)[torch.randperm(len(bucket_indices))].tolist()

            num_samples = len(bucket_indices)
            bs = cfg.batch_size

            for start in range(0, num_samples, bs):
                batch_indices = bucket_indices[start:start + bs]

                images = []
                captions = []

                for idx in batch_indices:
                    img_path, cap_path = dataset[idx]

                    text = open(cap_path, "r", encoding="utf-8").read().strip()

                    if cfg.memorize_first_token:
                        first = text.split(",")[0].strip()
                        if first:
                            text = f"{first}, {text}"

                    if cfg.prepend_token:
                        text = f"{cfg.prepend_token}, {text}"
                    if cfg.append_token:
                        text = f"{text}, {cfg.append_token}"

                    img = Image.open(img_path).convert("RGB")
                    images.append(tfm_bucket(img))
                    captions.append(text)

                pixel = torch.stack(images).to(device=device, dtype=dtype)

                if cfg.cache_latents:
                    assert cached_latents_by_bucket is not None
                    bucket_cache = cached_latents_by_bucket[bucket_res]
                    latents = torch.cat([bucket_cache[i] for i in batch_indices], dim=0).to(device=device, dtype=dtype)
                else:
                    with torch.no_grad():
                        latents = vae.encode(pixel).latent_dist.sample() * scaling_factor

                noise = torch.randn_like(latents)
                t = torch.randint(
                    0,
                    scheduler.config.num_train_timesteps,
                    (latents.size(0),),
                    device=device,
                ).long()

                noisy = scheduler.add_noise(latents, noise, t)

                with torch.no_grad():
                    prompt_embeds, pooled = encode_prompt_sdxl(pipe, captions, device, dtype)
                    add_time_ids = make_add_time_ids(latents.size(0), bucket_res, device, dtype)

                pred = unet(
                    noisy,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
                ).sample

                loss = F.mse_loss(pred.float(), noise.float())
                (loss / cfg.grad_accum_steps).backward()

                global_step += 1
                do_step = (global_step % cfg.grad_accum_steps == 0)

                if do_step:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    opt_step += 1

                    if opt_step % cfg.log_every == 0:
                        lr = lr_scheduler.get_last_lr()[0] if hasattr(lr_scheduler, "get_last_lr") else optimizer.param_groups[0]["lr"]
                        log(f"TRAIN epoch={epoch} opt_step={opt_step} lr={lr:.8f} loss={loss.item():.6f}")

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", choices=["sd", "sdxl"], required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--caption_ext", default=".txt")
    ap.add_argument("--prepend_token", default="")
    ap.add_argument("--append_token", default="")
    ap.add_argument("--resolution", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum_steps", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--cache_latents", action="store_true")
    ap.add_argument("--bucket", action="store_true")
    ap.add_argument("--bucket_min_res", type=int, default=512)
    ap.add_argument("--bucket_max_res", type=int, default=1536)
    ap.add_argument("--bucket_step", type=int, default=64)
    ap.add_argument("--scheduler_type", default="constant")
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--num_cycles", type=int, default=1)
    ap.add_argument("--lora_rank", type=int, default=8)
    ap.add_argument("--lora_alpha", type=float, default=8.0)
    ap.add_argument("--unet_lr", type=float, default=1e-4)
    ap.add_argument("--clip_lr", type=float, default=0.0)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--output", required=True)
    ap.add_argument("--save_every_epochs", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--do_inference", action="store_true")
    ap.add_argument("--inference_prompt", default="")
    ap.add_argument("--inference_steps", type=int, default=20)
    ap.add_argument("--inference_images", type=int, default=2)
    ap.add_argument("--clip_skip", type=int, default=0)
    ap.add_argument("--optimizer", default="adamw")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--epsilon", type=float, default=1e-8)
    ap.add_argument("--memorize_first_token", action="store_true")
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--nesterov", action="store_true")
    ap.add_argument("--target_modules", default="", help="Comma-separated list, e.g. to_q,to_k,to_v,to_out.0")
    ap.add_argument("--use_xformers", action="store_true", help="Enable xFormers memory efficient attention")
    ap.add_argument("--cpu_offload", action="store_true", help="Offload frozen components (VAE/text encoders) to CPU to save VRAM")

    args = ap.parse_args()

    cfg = TrainConfig(
        model_type=args.model_type,
        base_model=args.base_model,
        dataset=args.dataset,
        caption_ext=args.caption_ext,
        prepend_token=args.prepend_token.strip() or None,
        append_token=args.append_token.strip() or None,
        resolution=args.resolution,
        batch_size=args.batch_size,
        epochs=args.epochs,
        shuffle=args.shuffle,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        unet_lr=args.unet_lr,
        clip_lr=args.clip_lr,
        precision=args.precision,
        output=args.output,
        save_every_epochs=args.save_every_epochs,
        repeats=args.repeats,
        grad_accum_steps=args.grad_accum_steps,
        do_inference=args.do_inference,
        inference_prompt=args.inference_prompt,
        inference_steps=args.inference_steps,
        inference_images=args.inference_images,
        clip_skip=args.clip_skip,
        scheduler_type=args.scheduler_type,
        warmup_steps=args.warmup_steps,
        num_cycles=args.num_cycles,
        gradient_checkpointing=args.gradient_checkpointing,
        cache_latents=args.cache_latents,
        bucket_enabled=args.bucket,
        bucket_min_res=args.bucket_min_res,
        bucket_max_res=args.bucket_max_res,
        bucket_step=args.bucket_step,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        epsilon=args.epsilon,
        memorize_first_token=args.memorize_first_token,
        lora_dropout=args.lora_dropout,
        momentum=args.momentum,
        nesterov=args.nesterov,
        target_modules=parse_target_modules(args.target_modules),
        use_xformers=args.use_xformers,
        cpu_offload=args.cpu_offload,
    )

    train(cfg)

if __name__ == "__main__":
    main()
