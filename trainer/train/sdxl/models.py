import torch
from pathlib import Path
from safetensors.torch import load_file

from diffusers import UNet2DConditionModel, AutoencoderKL, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

from ..config import log

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "models"

SDXL_REF = "stabilityai/stable-diffusion-xl-base-1.0"

def resolve_model_identifier(model_id_or_path: str) -> str:
    raw = model_id_or_path.strip()
    p = Path(raw).expanduser()

    if p.exists():
        return str(p.resolve())

    candidate = MODELS_DIR / raw
    if candidate.exists():
        return str(candidate.resolve())

    return raw  # assume HF id

def detect_sdxl_safetensors_format(ckpt_path: Path) -> str:
    ckpt = load_file(str(ckpt_path), device="cpu")
    for k in ckpt.keys():
        if k.startswith("model.diffusion_model."):
            return "merged"
        if k.startswith("unet.") or k.startswith("vae.") or k.startswith("text_encoder."):
            return "diffusers"
    raise RuntimeError(f"Unrecognized SDXL safetensors format: {ckpt_path.name}")

def _load_sdxl_components_diffusers(base: str, device, dtype):
    unet = UNet2DConditionModel.from_pretrained(base, subfolder="unet", torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=dtype).to(device)
    te1 = CLIPTextModel.from_pretrained(base, subfolder="text_encoder", torch_dtype=dtype).to(device)
    te2 = CLIPTextModel.from_pretrained(base, subfolder="text_encoder_2", torch_dtype=dtype).to(device)

    tok1 = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(base, subfolder="tokenizer_2")

    log("STATUS all SDXL components loaded (diffusers)")
    return unet, vae, te1, te2, tok1, tok2

def _load_sdxl_components_safetensors_diffusers(ckpt_path: Path, device, dtype):
    log("STATUS detected diffusers-style SDXL safetensors")

    unet = UNet2DConditionModel.from_pretrained(SDXL_REF, subfolder="unet", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(SDXL_REF, subfolder="vae", torch_dtype=dtype)
    te1 = CLIPTextModel.from_pretrained(SDXL_REF, subfolder="text_encoder", torch_dtype=dtype)
    te2 = CLIPTextModel.from_pretrained(SDXL_REF, subfolder="text_encoder_2", torch_dtype=dtype)

    tok1 = CLIPTokenizer.from_pretrained(SDXL_REF, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(SDXL_REF, subfolder="tokenizer_2")

    ckpt = load_file(str(ckpt_path), device="cpu")

    unet.load_state_dict({k[5:]: v for k, v in ckpt.items() if k.startswith("unet.")}, strict=True)
    vae.load_state_dict({k[4:]: v for k, v in ckpt.items() if k.startswith("vae.")}, strict=True)
    te1.load_state_dict({k[13:]: v for k, v in ckpt.items() if k.startswith("text_encoder.")}, strict=True)
    te2.load_state_dict({k[15:]: v for k, v in ckpt.items() if k.startswith("text_encoder_2.")}, strict=True)

    unet.to(device=device, dtype=dtype)
    vae.to(device=device, dtype=dtype)
    te1.to(device=device, dtype=dtype)
    te2.to(device=device, dtype=dtype)

    log("STATUS all SDXL components loaded (diffusers safetensors)")
    return unet, vae, te1, te2, tok1, tok2

def _load_sdxl_components_safetensors_merged(ckpt_path: Path, device, dtype):
    log("STATUS detected merged SDXL safetensors")

    unet = UNet2DConditionModel.from_pretrained(SDXL_REF, subfolder="unet", torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(SDXL_REF, subfolder="vae", torch_dtype=dtype)
    te1 = CLIPTextModel.from_pretrained(SDXL_REF, subfolder="text_encoder", torch_dtype=dtype)
    te2 = CLIPTextModel.from_pretrained(SDXL_REF, subfolder="text_encoder_2", torch_dtype=dtype)

    tok1 = CLIPTokenizer.from_pretrained(SDXL_REF, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(SDXL_REF, subfolder="tokenizer_2")

    ckpt = load_file(str(ckpt_path), device="cpu")

    unet_sd, vae_sd, te1_sd, te2_sd = {}, {}, {}, {}
    for k, v in ckpt.items():
        if k.startswith("model.diffusion_model."):
            unet_sd[k.replace("model.diffusion_model.", "")] = v
        elif k.startswith("first_stage_model."):
            vae_sd[k.replace("first_stage_model.", "")] = v
        elif k.startswith("conditioner.embedders.0."):
            te1_sd[k.replace("conditioner.embedders.0.", "")] = v
        elif k.startswith("conditioner.embedders.1."):
            te2_sd[k.replace("conditioner.embedders.1.", "")] = v

    unet.load_state_dict(unet_sd, strict=False)
    vae.load_state_dict(vae_sd, strict=False)
    te1.load_state_dict(te1_sd, strict=False)
    te2.load_state_dict(te2_sd, strict=False)

    unet.to(device=device, dtype=dtype)
    vae.to(device=device, dtype=dtype)
    te1.to(device=device, dtype=dtype)
    te2.to(device=device, dtype=dtype)

    log("STATUS all SDXL components loaded (merged safetensors)")
    return unet, vae, te1, te2, tok1, tok2

def load_sdxl_components(base_model: str, device, dtype):
    ident = resolve_model_identifier(base_model)

    if ident.endswith(".safetensors"):
        p = Path(ident)
        if not p.is_file():
            raise FileNotFoundError(f"SDXL checkpoint not found: {ident}")

        fmt = detect_sdxl_safetensors_format(p)
        if fmt == "diffusers":
            return _load_sdxl_components_safetensors_diffusers(p, device, dtype)
        if fmt == "merged":
            return _load_sdxl_components_safetensors_merged(p, device, dtype)

    return _load_sdxl_components_diffusers(ident, device, dtype)


def load_sdxl_scheduler(base_model: str):
    ident = resolve_model_identifier(base_model)
    return DDPMScheduler.from_pretrained(
        SDXL_REF if ident.endswith(".safetensors") else ident,
        subfolder="scheduler",
    )
