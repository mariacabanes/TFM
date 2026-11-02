#!/bin/bash
# =============================================================================
#  setup_env.sh  —  One-shot environment setup for UNet / FL training
#  Usage:  bash setup_env.sh
# =============================================================================

set -e  # stop immediately if any command fails

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[91m'; GREEN='\033[92m'; YELLOW='\033[93m'
CYAN='\033[96m'; BOLD='\033[1m'; RESET='\033[0m'

ok()   { echo -e "${GREEN}[OK]  $1${RESET}"; }
info() { echo -e "${CYAN}[i]   $1${RESET}"; }
warn() { echo -e "${YELLOW}[!]   $1${RESET}"; }
err()  { echo -e "${RED}[X]   $1${RESET}"; exit 1; }
step() { echo -e "\n${BOLD}${CYAN}──────────────────────────────────────────${RESET}"; \
         echo -e "${BOLD}${CYAN}  $1${RESET}"; \
         echo -e "${BOLD}${CYAN}──────────────────────────────────────────${RESET}"; }

# ── Config ────────────────────────────────────────────────────────────────────
VENV_DIR="./envs/fl_env"
PYTHON="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"

# =============================================================================
step "1/6  System packages"
# =============================================================================

info "Updating apt..."
sudo apt update -qq

info "Installing python3, pip, venv, dev headers..."
sudo apt install -y python3 python3-pip python3-venv python3-dev > /dev/null
ok "Python packages installed."

info "Installing NVIDIA CUDA toolkit..."
if sudo apt install -y nvidia-cuda-toolkit > /dev/null 2>&1; then
    ok "CUDA toolkit installed."
else
    warn "CUDA toolkit install had issues — continuing anyway."
    warn "If nvidia-smi works your driver is fine; PyTorch will still use the GPU."
fi

# =============================================================================
step "2/6  Creating virtual environment"
# =============================================================================

if [ -d "$VENV_DIR" ]; then
    warn "Virtual environment already exists at $VENV_DIR — skipping creation."
else
    python3 -m venv "$VENV_DIR"
    ok "Virtual environment created at $VENV_DIR"
fi

# =============================================================================
step "3/6  Upgrading pip"
# =============================================================================

"$PIP" install --upgrade pip --quiet
ok "pip upgraded to $($PIP --version | awk '{print $2}')"

# =============================================================================
step "4/6  Installing PyTorch with CUDA 12.1"
# =============================================================================

info "This may take a few minutes (~3-5 GB download)..."
"$PIP" install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --quiet
ok "PyTorch installed."

# =============================================================================
step "5/6  Installing training dependencies"
# =============================================================================

"$PIP" install \
    pycocotools \
    scipy \
    matplotlib \
    tqdm \
    numpy \
    dill \
    blosc \
    torch_geometric \
    pandas \
    seaborn \
    scikit-learn \
    IPython \
    opencv-contrib-python \
    --quiet
ok "Training dependencies installed."

# =============================================================================
step "6/6  Verifying GPU and saving requirements"
# =============================================================================

info "Checking CUDA availability..."
"$PYTHON" - <<'EOF'
import torch
cuda_ok = torch.cuda.is_available()
if cuda_ok:
    print(f"\033[92m[OK]  CUDA available : True\033[0m")
    print(f"\033[92m[OK]  GPU detected   : {torch.cuda.get_device_name(0)}\033[0m")
    print(f"\033[92m[OK]  CUDA version   : {torch.version.cuda}\033[0m")
else:
    print(f"\033[93m[!]   CUDA available : False\033[0m")
    print(f"\033[93m[!]   Training will run on CPU only.\033[0m")
    print(f"\033[93m[!]   Check your NVIDIA driver with: nvidia-smi\033[0m")
EOF

info "Saving requirements.txt..."
"$PIP" freeze > requirements.txt
ok "requirements.txt saved."

# =============================================================================
step "7/7  Creating input_zips directory"
# =============================================================================

mkdir -p input_zips
ok "input_zips directory created."

# =============================================================================
echo ""
echo -e "${BOLD}${GREEN}══════════════════════════════════════════${RESET}"
echo -e "${BOLD}${GREEN}  Setup complete!${RESET}"
echo -e "${BOLD}${GREEN}══════════════════════════════════════════${RESET}"
echo ""
echo -e "  To activate the environment in a new terminal:"
echo -e "  ${CYAN}source $VENV_DIR/bin/activate${RESET}"

echo -e "  To run the pipeline:"
echo -e "  ${CYAN}python '(0) run_all.py'${RESET}"
