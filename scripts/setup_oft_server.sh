#!/usr/bin/env bash
# Set up a venv for the OpenVLA-OFT (L1 regression + action head + 2 cams) policy server.
# Resulting venv: ./_envs/openvla-oft/.venv
#
# Usage:
#   ./scripts/setup_oft_server.sh
#
# Requires: uv, git, NVIDIA GPU with CUDA >= 12.8.
set -euo pipefail

HERE=$(cd "$(dirname "$0")"/.. && pwd)
ENV_ROOT="${ENV_ROOT:-$HERE/_envs}"
mkdir -p "$ENV_ROOT"
cd "$ENV_ROOT"

if [[ ! -d openvla-oft ]]; then
  git clone https://github.com/moojink/openvla-oft.git
fi
cd openvla-oft

uv venv --python 3.10 .venv
export VIRTUAL_ENV="$PWD/.venv"

uv pip install -e .

# Blackwell support
uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

uv pip install "numpy<2" "ml_dtypes<0.4"

# Apply our TROSSEN_CONSTANTS + forge_dataset registration
# (constants.py + configs.py + transforms.py + mixtures.py)
git apply "$HERE/patches/openvla_oft.patch"

uv pip install "$HERE/packages/openpi-client"
uv pip install "$HERE"

echo
echo "Done. To use this venv:"
echo "    source $PWD/.venv/bin/activate"
echo "    trossen-serve-oft --checkpoint UoA-Trossen-Arm/openvla-7b-oft-lift-eggplant --port 8000"
