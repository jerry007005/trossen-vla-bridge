#!/bin/bash --login
#SBATCH --job-name=smoke-oft-6task
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --nodelist=porcini
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/node-shared/jerry007005/haochuan/trossen_vla_bridge/slurm_logs/%x-%j.out
#SBATCH --error=/node-shared/jerry007005/haochuan/trossen_vla_bridge/slurm_logs/%x-%j.err

set -euo pipefail
[[ -f ~/.bashrc ]] && source ~/.bashrc

mkdir -p /node-shared/jerry007005/haochuan/trossen_vla_bridge/slurm_logs

export HF_HOME=/node-shared/jerry007005/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers

cd /node-shared/jerry007005/haochuan/openvla-oft
source .venv/bin/activate

echo "Node: $(hostname)"
nvidia-smi -L

cd /node-shared/jerry007005/haochuan/trossen_vla_bridge
python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, "src")
# serve_oft.py appends '--robot=trossen' to argv at import time to force
# TROSSEN_CONSTANTS; that path runs here too.
from trossen_vla_bridge.serve_oft import OpenVLAOFTPolicy

print(">>> instantiating OpenVLAOFTPolicy with the new bridge defaults")
policy = OpenVLAOFTPolicy(
    checkpoint="UoA-Trossen-Arm/openvla-7b-oft-trossen-6task",
    unnorm_key="trossen_6task_combined",
)

print("\n>>> sending one synthetic obs through .infer for each of the 6 prompts")
prompts = [
    "lift the eggplant",
    "lift the carrot",
    "pick up the carrot and place it in the plate",
    "lift the pineapple",
    "pick up the eggplant and place it in the plate",
    "pick up the pineapple and place it in the plate",
]
state = np.zeros(7, dtype=np.float32)
img = np.zeros((3, 224, 224), dtype=np.uint8)
for p in prompts:
    out = policy.infer({"state": state, "images": {"cam_main": img, "cam_wrist": img}, "prompt": p})
    a = out["actions"]
    print(f"  prompt={p!r:60s} -> actions shape={a.shape} dtype={a.dtype} "
          f"range=[{a.min():+.3f}, {a.max():+.3f}]")
    assert a.shape == (20, 7), f"expected (20, 7), got {a.shape}"
    assert a.dtype == np.float32
    assert np.isfinite(a).all(), "non-finite action values"

print("\n[oft] smoke test PASSED — model loads, infers, returns (20, 7) finite float32 actions for all 6 prompts.")
PY
