- LoRA Trainer (SD 1.5 & SDXL)
- A simple web UI for training LoRA models.
- Designed to avoid overwhelming settings and focus on the ones that actually matter.
- Supports SD 1.5 and SDXL only
- Minimal UI, safety checks included
- No advanced / experimental features by default
- Supports Windows 11 and Linux (Tested on Ubuntu 24.04 & Windows 11) 
----------------------------
System Requirements:
###
1) NVIDIA GPU (required for training)
2) CUDA compatible with your PyTorch install
----------------------------
- Python 3.10+ recommended
----------------------------
- Conda environment strongly suggested
----------------------------
Installation:
###
1) git clone https://github.com/bravooscarromeo/simple-lora-trainer.git
2) cd path/to/Lora_Trainer
3) pip install -r requirements.txt
4) pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuNNN (Specify version in for NNN) (Must be compatible with Toolkit version)
5) python app.py
6) hosted on port 5000 - ctrl + click the link in terminal
----------------------------
Optional (Recommended) - 
###
- xFormers (must match your PyTorch + CUDA versions)
-----------------------------
Usage - 
###
1) Create a project (folder defaults to home/yourusername/lora_project)
2) Configure settings
3) Click Train
4) Go make some coffee and maybe mow the lawn too while you wait 
----------------------------

