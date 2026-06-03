#!/usr/bin/env bash
# Set up a venv for the pi0 (PyTorch) policy server.
# Resulting venv: ./_envs/openpi/.venv
#
# Usage:
#   ./scripts/setup_pi0_server.sh
#
# Requires: uv, git, NVIDIA GPU with CUDA >= 12.8.
set -euo pipefail

HERE=$(cd "$(dirname "$0")"/.. && pwd)
ENV_ROOT="${ENV_ROOT:-$HERE/_envs}"
mkdir -p "$ENV_ROOT"
cd "$ENV_ROOT"

if [[ ! -d openpi ]]; then
  git clone https://github.com/Physical-Intelligence/openpi.git
fi
cd openpi
git submodule update --init --recursive

uv venv --python 3.11 .venv
export VIRTUAL_ENV="$PWD/.venv"
export UV_PROJECT_ENVIRONMENT="$PWD/.venv"

# uv sync pulls in jax, lerobot, etc.
GIT_LFS_SKIP_SMUDGE=1 uv sync

# Blackwell support: openpi pins torch 2.7.1 which lacks sm_120 binaries.
uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# pi0's PyTorch implementation patches transformers in-place; required to load
# the model. WARNING: this permanently mutates the transformers package in your
# uv cache. To undo: `uv cache clean transformers`.
cp -r src/openpi/models_pytorch/transformers_replace/* \
      .venv/lib/python3.11/site-packages/transformers/

# Switch LeRobot video decoder to pyav so we don't need a system ffmpeg.
uv pip install av

# Apply our config.py + data_loader.py + trossen_policy.py for the Trossen
# lift-eggplant train-config and DataConfig.
git apply "$HERE/patches/openpi.patch"
cp "$HERE/patches/openpi_trossen_policy.py" src/openpi/policies/trossen_policy.py

uv pip install "$HERE/packages/openpi-client"
uv pip install "$HERE"

echo
echo "Done. To use this venv:"
echo "    source $PWD/.venv/bin/activate"
echo "    trossen-serve-pi0 --checkpoint UoA-Trossen-Arm/pi0-lift-eggplant-pytorch --port 8000"
