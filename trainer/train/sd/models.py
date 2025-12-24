import torch
from pathlib import Path
from safetensors.torch import load_file

from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from ..config import TrainConfig, log

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models"

SD15_REF = "runwayml/stable-diffusion-v1-5"

def resolve_model_identifier(model_id_or_path: str) -> str:
    raw = model_id_or_path.strip()
    p = Path(raw).expanduser()

    if p.exists():
        return str(p.resolve())

    candidate = MODELS_DIR / raw
    if candidate.exists():
        return str(candidate.resolve())

    return raw

def _load_sd_models_diffusers(base: str, device, dtype):
    tok = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=dtype).to(device)
    sched = DDPMScheduler.from_pretrained(base, subfolder="scheduler")
    return tok, te, vae, unet, sched

def _load_sd_models_safetensors(ckpt_path: Path, device, dtype):
    log("STATUS loading merged SD checkpoint (.safetensors)")

    tok = CLIPTokenizer.from_pretrained(SD15_REF, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(SD15_REF, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(SD15_REF, subfolder="vae", torch_dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(SD15_REF, subfolder="unet", torch_dtype=dtype)
    sched = DDPMScheduler.from_pretrained(SD15_REF, subfolder="scheduler")

    ckpt = load_file(str(ckpt_path), device="cpu")

    unet_sd, vae_sd, te_sd = {}, {}, {}
    for k, v in ckpt.items():
        if k.startswith("model.diffusion_model."):
            unet_sd[k.replace("model.diffusion_model.", "")] = v
        elif k.startswith("first_stage_model."):
            vae_sd[k.replace("first_stage_model.", "")] = v
        elif k.startswith("cond_stage_model."):
            te_sd[k.replace("cond_stage_model.", "")] = v

    unet.load_state_dict(unet_sd, strict=False)
    vae.load_state_dict(vae_sd, strict=False)
    te.load_state_dict(te_sd, strict=False)

    unet.to(device=device, dtype=dtype)
    vae.to(device=device, dtype=dtype)
    te.to(device=device, dtype=dtype)

    log("STATUS all SD components loaded (safetensors)")
    return tok, te, vae, unet, sched

def load_sd_models(cfg: TrainConfig, device, dtype):
    ident = resolve_model_identifier(cfg.base_model)

    if ident.endswith(".safetensors"):
        p = Path(ident)
        if not p.is_file():
            raise FileNotFoundError(f"SD checkpoint not found: {ident}")
        tok, te, vae, unet, sched = _load_sd_models_safetensors(p, device, dtype)
    else:
        tok, te, vae, unet, sched = _load_sd_models_diffusers(ident, device, dtype)

    if cfg.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        log("STATUS gradient_checkpointing=ENABLED")
    else:
        log("STATUS gradient_checkpointing=DISABLED")

    return tok, te, vae, unet, sched
