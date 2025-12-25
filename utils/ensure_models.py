from pathlib import Path

from trainer.train.config import log
from utils.paths import MODELS_DIR
from trainer.train.sd.models import resolve_model_identifier


HF_CACHE_ROOT = MODELS_DIR / "hf_cache"


def _hf_repo_dir_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def ensure_base_model_available(model_id: str):
    ident = resolve_model_identifier(model_id)

    if Path(ident).exists():
        return

    base_dir = MODELS_DIR / "base" / model_id
    if base_dir.exists():
        return

    repo_dir = _hf_repo_dir_name(model_id)

    if (
        (HF_CACHE_ROOT / "hub" / repo_dir).exists()
        or (HF_CACHE_ROOT / "transformers" / repo_dir).exists()
    ):
        return

    log("=" * 80)
    log("⚠️  FIRST RUN: BASE MODEL NOT FOUND")
    log(f"⚠️  Downloading model '{model_id}'")
    log("⚠️  This can take several minutes depending on your connection.")
    log("=" * 80)
