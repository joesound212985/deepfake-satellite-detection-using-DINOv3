#!/bin/bash
# Create / update the python environment for the GRSL experiments (new machine).
#   ./setup_env.sh            -> conda env "torch313" (created if missing), CUDA torch, all deps
#   CUDA=cu128 ./setup_env.sh -> pick a torch CUDA build (cu128 default: works for RTX 40xx and 50xx)
set -e
ENV=${ENV:-torch313}
CUDA=${CUDA:-cu128}
if ! command -v conda >/dev/null; then echo "conda not found; install Miniconda first"; exit 1; fi
eval "$(conda shell.bash hook)"
conda env list | grep -q "^$ENV " || conda create -n "$ENV" python=3.13 -y
conda activate "$ENV"
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/$CUDA
pip install -q "transformers>=5" timm opencv-python-headless scikit-learn matplotlib tqdm packaging "huggingface_hub[cli]" pillow pandas requests tifffile einops open_clip_torch lightning wandb kornia torchmetrics
python - <<'EOF'
import torch, transformers, timm, cv2, sklearn
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__, "timm", timm.__version__, "opencv", cv2.__version__, "sklearn", sklearn.__version__)
EOF
echo "now run:  hf auth login   (then ./setup_data.sh)"
