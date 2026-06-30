#!/bin/bash --login
#SBATCH --job-name=openvla-before-after
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
export HF_HOME=/node-shared/jerry007005/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers

cd /node-shared/jerry007005/haochuan/openvla
source .venv/bin/activate
echo "Node: $(hostname)"; nvidia-smi -L
cd /node-shared/jerry007005/haochuan/trossen_vla_bridge

python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, "src")
from trossen_vla_bridge.serve_openvla import OpenVLAPolicy

f = np.load("scripts/_real_frame_two.npz")
state, gt_action = f["state"], f["action_gt"]
gt_delta = gt_action - state
main_224 = f["cam_main_224"]
main_raw = f["cam_main_raw"]

print("loading vanilla OpenVLA policy ...")
policy = OpenVLAPolicy(
    checkpoint="UoA-Trossen-Arm/openvla-7b-trossen-6task",
    unnorm_key="trossen_6task_combined",
)

def run(label, main_chw):
    # do_sample=False already in serve_openvla, so deterministic
    out = policy.infer({
        "state": state,
        "images": {"cam_main": main_chw},  # vanilla = primary only
        "prompt": "lift the eggplant",
    })
    delta = out["actions"][0] - state
    err = float(np.abs(delta - gt_delta).sum())
    print(f"  {label:30s} L1={err:.4f}  delta=[{', '.join(f'{x:+.4f}' for x in delta)}]")
    return err

print(f"\nGT step-0 delta: {[f'{x:+.4f}' for x in gt_delta]}  L1={float(np.abs(gt_delta).sum()):.4f}\n")
print("vanilla OpenVLA argmax-decodes the 7 discrete action tokens (do_sample=False),\n"
      "so 1 run each is enough.\n")
print("=== BEFORE 5cadbe2 (cv2-resized 224x224) ===")
e_before = run("cv2 224x224", main_224)
print("\n=== AFTER 5cadbe2 (raw 480x640) ===")
e_after  = run("raw 480x640", main_raw)
print(f"\n*** L1 change: {e_before:.4f} -> {e_after:.4f}  ({(e_after - e_before) / e_before * 100:+.1f}%)")
PY
