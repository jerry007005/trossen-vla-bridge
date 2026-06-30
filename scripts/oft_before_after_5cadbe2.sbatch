#!/bin/bash --login
#SBATCH --job-name=oft-before-after
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

cd /node-shared/jerry007005/haochuan/openvla-oft
source .venv/bin/activate
echo "Node: $(hostname)"; nvidia-smi -L
cd /node-shared/jerry007005/haochuan/trossen_vla_bridge

python3 - <<'PY'
import sys, numpy as np
sys.path.insert(0, "src")
from trossen_vla_bridge.serve_oft import OpenVLAOFTPolicy

f = np.load("scripts/_real_frame_two.npz")
state, gt_action = f["state"], f["action_gt"]
gt_delta = gt_action - state
main_224, wrist_224 = f["cam_main_224"], f["cam_wrist_224"]
main_raw, wrist_raw = f["cam_main_raw"], f["cam_wrist_raw"]

print("loading OFT policy ...")
policy = OpenVLAOFTPolicy(
    checkpoint="UoA-Trossen-Arm/openvla-7b-oft-trossen-6task",
    unnorm_key="trossen_6task_combined",
)

def run(label, main_chw, wrist_chw):
    out = policy.infer({
        "state": state,
        "images": {"cam_main": main_chw, "cam_wrist": wrist_chw},
        "prompt": "lift the eggplant",
    })
    delta = out["actions"][0] - state
    err = float(np.abs(delta - gt_delta).sum())
    print(f"  {label:30s} L1={err:.4f}  delta=[{', '.join(f'{x:+.4f}' for x in delta)}]")
    return err

print(f"\nGT step-0 delta: {[f'{x:+.4f}' for x in gt_delta]}  L1={float(np.abs(gt_delta).sum()):.4f}\n")
print("OFT is deterministic (L1 regression head, no sampling), so 1 run each is enough.\n")
print("=== BEFORE 5cadbe2 (cv2-resized 224x224) ===")
e_before = run("cv2 224x224", main_224, wrist_224)
print("\n=== AFTER 5cadbe2 (raw 480x640) ===")
e_after  = run("raw 480x640", main_raw, wrist_raw)
print(f"\n*** L1 change: {e_before:.4f} -> {e_after:.4f}  ({(e_after - e_before) / e_before * 100:+.1f}%)")
PY
