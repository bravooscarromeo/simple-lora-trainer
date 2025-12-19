from pathlib import Path
import subprocess
import shlex
import yaml
import os
from safetensors import safe_open
from utils.trainer_cli_adapter import build_train_lora_cli_args


BASE_DIR = Path.home() / "lora_projects"

class TrainingConfigError(Exception):
    """Fatal configuration error that should never occur if Save validation is correct."""
    pass

def launch_training(project_name: str):
    project_dir = BASE_DIR / project_name
    config_path = project_dir / "config.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    log_dir = project_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    config = yaml.safe_load(config_path.read_text())

    model = config.get("model", {})
    arch = model.get("architecture")

    if not arch:
        raise TrainingConfigError("config.model.architecture is required")
    
    if arch in ("sd15", "sd1", "sd1.5", "sd"):
        model_type = "sd"
    elif arch == "sdxl":
        model_type = "sdxl"
    else:
        raise TrainingConfigError(
            f"Invalid model architecture '{arch}'. Expected sd or sdxl."
        )

    # --- Validate dataset path ---
    dataset = config["dataset"]
    train_data_dir = project_dir / dataset["path"]
    if not train_data_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {train_data_dir}")

    output_dir = project_dir / "output"
    output_dir.mkdir(exist_ok=True)

    args = build_train_lora_cli_args(config, project_dir)

    trainer_dir = Path(__file__).resolve().parent.parent / "trainer"

    if model_type == "sd":
        TRAINER = trainer_dir / "train_lora_v1.py"
    elif model_type == "sdxl":
        TRAINER = trainer_dir / "train_lora_sdxl_v1.py"
    else:
        raise TrainingConfigError(f"Unsupported model_type: {model_type}")

    if not TRAINER.exists():
        raise FileNotFoundError(f"Trainer not found: {TRAINER}")

    cmd = ["python", str(TRAINER), *args]

    print("[TRAIN] Launching training:")
    print(" ".join(shlex.quote(c) for c in cmd))

    process = subprocess.Popen(
        cmd,
        stdout=open(log_dir / "train.log", "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    pid_file = project_dir / "training.pid"
    pgid = os.getpgid(process.pid)
    pid_file.write_text(str(pgid))

    print(f"[TRAIN] Training started (PGID {pgid})")
