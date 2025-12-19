from pathlib import Path

BASE_DIR = Path.home() / "lora_projects"
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)