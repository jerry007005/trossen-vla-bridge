#!/bin/bash --login
#SBATCH --job-name=cmp-oft-realframe
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

echo "Node: $(hostname)"
nvidia-smi -L
cd /node-shared/jerry007005/haochuan/trossen_vla_bridge

python3 - <<'PY'
import json, sys, numpy as np
sys.path.insert(0, "src")
from trossen_vla_bridge.serve_oft import OpenVLAOFTPolicy

frame = np.load("scripts/_real_frame.npz", allow_pickle=True)
state, gt_action = frame["state"], frame["action_gt"]
prompt = str(frame["prompt"])
print(f"frame: prompt={prompt!r}, state={state.tolist()}")

policy = OpenVLAOFTPolicy(
    checkpoint="UoA-Trossen-Arm/openvla-7b-oft-trossen-6task",
    unnorm_key="trossen_6task_combined",
)
out = policy.infer({
    "state": state,
    "images": {"cam_main": frame["cam_main"], "cam_wrist": frame["cam_wrist"]},
    "prompt": prompt,
})
a = out["actions"]
delta = a[0] - state
gt_delta = gt_action - state
print(f"oft chunk shape: {a.shape}")
print(f"oft step-0 abs : {a[0].tolist()}")
print(f"oft step-0 dl  : {delta.tolist()}")
print(f"GT     dl      : {gt_delta.tolist()}")
print(f"L1 err vs GT   : {float(np.abs(delta - gt_delta).sum()):.4f}")

json.dump(
    {
        "model": "oft",
        "abs_step0": a[0].tolist(),
        "delta_step0": delta.tolist(),
        "abs_full": a.tolist(),
        "l1_err_step0": float(np.abs(delta - gt_delta).sum()),
    },
    open("scripts/_cmp_oft.json", "w"),
    indent=2,
)
print("wrote scripts/_cmp_oft.json")
PY
