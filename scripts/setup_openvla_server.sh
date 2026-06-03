#!/usr/bin/env bash
# Set up a venv for the OpenVLA-7B (autoregressive) policy server.
# Resulting venv: ./_envs/openvla/.venv
#
# Usage:
#   ./scripts/setup_openvla_server.sh
#
# Requires: uv (https://docs.astral.sh/uv/), git, NVIDIA GPU with CUDA >= 12.8 (for Blackwell support).
set -euo pipefail

HERE=$(cd "$(dirname "$0")"/.. && pwd)
ENV_ROOT="${ENV_ROOT:-$HERE/_envs}"
mkdir -p "$ENV_ROOT"
cd "$ENV_ROOT"

if [[ ! -d openvla ]]; then
  git clone https://github.com/openvla/openvla.git
fi
cd openvla

uv venv --python 3.10 .venv
export VIRTUAL_ENV="$PWD/.venv"

# Base openvla install (pins torch 2.2 which lacks Blackwell sm_120 support).
uv pip install -e .

# Blackwell support: upgrade torch to cu128 wheel.
uv pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# tf 2.15 ships only numpy<2 wheels.
uv pip install "numpy<2" "ml_dtypes<0.4"

# Apply our forge_dataset registration (configs.py + transforms.py).
git apply "$HERE/patches/openvla.patch"

# Vendored openpi-client (websocket protocol library, with keepalive disabled).
uv pip install "$HERE/packages/openpi-client"

# Install this bridge package so `trossen-serve-openvla` is on PATH.
uv pip install "$HERE"

echo
echo "Done. To use this venv:"
echo "    source $PWD/.venv/bin/activate"
echo "    trossen-serve-openvla --checkpoint UoA-Trossen-Arm/openvla-7b-lift-eggplant --port 8000"
