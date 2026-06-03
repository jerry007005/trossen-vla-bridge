#!/usr/bin/env bash
# Set up the hardware-side venv that talks to a Trossen AI Solo arm
# (the policy server can be running on any machine reachable by network).
# Resulting venv: ./_envs/lerobot/.venv
#
# Usage:
#   ./scripts/setup_client.sh
#
# Requires: uv, git, a Linux machine with USB access to the Trossen arm + RealSense cameras.
set -euo pipefail

HERE=$(cd "$(dirname "$0")"/.. && pwd)
ENV_ROOT="${ENV_ROOT:-$HERE/_envs}"
mkdir -p "$ENV_ROOT"
cd "$ENV_ROOT"

if [[ ! -d lerobot ]]; then
  git clone -b trossen-ai https://github.com/Interbotix/lerobot.git
fi
cd lerobot

# Use a uv-managed Python so evdev (deep dep of pynput) can find Python.h.
uv venv --python 3.10 .venv
export VIRTUAL_ENV="$PWD/.venv"

uv pip install -e ".[trossen_ai]"

# Update TrossenAISoloRobotConfig to the deployed arm's IP + camera SNs.
# Edit the patch before applying if your hardware differs.
git apply "$HERE/patches/lerobot.patch"

uv pip install "$HERE/packages/openpi-client"
uv pip install "$HERE"

echo
echo "Done. To use this venv:"
echo "    source $PWD/.venv/bin/activate"
echo
echo "Step A — hardware connectivity sanity check (no policy server, no motion):"
echo "    trossen-client --connect-only"
echo
echo "Step B — dry-run with a policy server (reads obs + queries server, no motion):"
echo "    trossen-client --mode test --task-prompt \"lift the eggplant\" --server-host <host> --server-port 8000"
echo
echo "Step C — autonomous (real motion; have the E-stop ready):"
echo "    trossen-client --mode autonomous --task-prompt \"lift the eggplant\" --server-host <host> --server-port 8000"
