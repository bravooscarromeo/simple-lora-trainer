- LoRA Trainer (SD 1.5 & SDXL)
- A simple web UI for training LoRA models using sd-scripts.
- Designed to avoid overwhelming settings and focus on the ones that actually matter.
- Supports SD 1.5 and SDXL only
- For Linux (created & tested on Ubuntu 24.04 LTS)
- Minimal UI, safety checks included
- No advanced / experimental features by default
- This project is considered feature-complete.
- Only bug fixes will be added.
----------------------------
 - System Requirements:
1) Linux (Ubuntu recommended)
2) NVIDIA GPU (required for training)
3) CUDA compatible with your PyTorch install
----------------------------
- Python - 
- Python 3.10+ recommended
----------------------------
- Conda environment strongly suggested
----------------------------
Installation:
1) git clone --recursive https://github.com/bravooscarromeo/simple-lora-trainer.git
2) cd path/to/Lora_Trainer
3) pip install -r requirements.txt
4) pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuNNN (Specify version in for NNN)
###
Training Backend
This app uses sd-scripts.
You must install its dependencies manually:
###
5) cd path/to/Lora_Trainer/trainer/sd-scripts
6) pip install -r requirements.txt
7) pip install accelerate
8) accelerate config
9) cd path/to/Lora_Trainer
10) python app.py
11) hosted on port 5000 - ctrl + click link in terminal
----------------------------
- Optional - 
- xFormers (must match your PyTorch + CUDA versions)
-----------------------------
Usage
1) Create a project (folder defaults to home/yourusername/lora_project
- !! Place your image folders inside the project’s dataset/ directory
- !! Do not place images directly in dataset/ — use a subfolder per project
2) Configure settings
3) Click Train
----------------------------
The UI includes:
Warnings for risky settings
Hard stops for invalid configurations
