from pathlib import Path
import yaml

BASE_DIR = Path.home() / "lora_projects"

def build_default_config(project_name: str):
    return {
        "project": {
            "name": project_name,
            "description": ""
        },

        "model": {
            "architecture": "sdxl",
            "checkpoint": None
        },

        "dataset": {
            "path": "dataset/",
            "resolution": 1024,
            "repeats": 10,
            "batch_size": 1,
            "shuffle": True,
            "captions": {
                "extension": ".txt",
                "first_word_memorize": False,
                "prepend_token": "",
                "append_token": ""
            },
            "cache_latents": True,
            "bucket": {
                "enabled": True,
                "min_res": 512,
                "max_res": 1536,
                "step": 64
            }
        },

        "training": {
            "epochs": 10,
            "save_every_epochs": 1,
            "do_inference": True, 
            "gradient_accumulation": 1,
            "conditioning": {
                "clip_skip": 1
            },
            "learning_rates": {
                "unet": 2e-05,
                "clip": 2e-05,
            }
        },

        "lora": {
            "rank": 64,
            "alpha": 64,
            "dropout": 0.0,
            "target_modules": "auto",
        },

        "optimizer": {
            "type": "adamw",
            "weight_decay": 0.01,
            "betas": [0.9, 0.999],
            "epsilon": 1e-08
        },

        "scheduler": {
            "type": "cosine",
            "warmup_steps": 0,
            "num_cycles": 1
        },

        "vae": {
            "path": None,
            "bake_into_latents": True,
            "use_for_sampling": True
        },

        "precision": {
            "mixed_precision": "fp16",
            "gradient_checkpointing": False,
            "xformers": False,
            "cpu_offload": False
        },

        "logging": {
            "log_interval": 10,
            "tensorboard": True,
            "save_samples": True
        },

        "output": {
            "save_state": False,
            "save_format": "safetensors"
        }
    }

def create_project(name: str):
    project_dir = BASE_DIR / name
    project_dir.mkdir(parents=True, exist_ok=False)

    (project_dir / "dataset").mkdir(exist_ok=True)
    (project_dir / "output").mkdir(exist_ok=True)
    (project_dir / "logs").mkdir(exist_ok=True)

    config = build_default_config(name)
    config_path = project_dir / "config.yaml"
    config_path.write_text(yaml.dump(config, sort_keys=False))

    return project_dir

if __name__ == "__main__":
    name = input("Project name: ").strip()
    if name:
        path = create_project(name)
        print(f"Project created at: {path}")
