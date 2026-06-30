#!/bin/bash --login
#SBATCH --job-name=pi0-before-after
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

export HF_LEROBOT_HOME=/node-shared/jerry007005/haochuan/lerobot_home
export HF_HOME=/node-shared/jerry007005/huggingface
export OPENPI_DATA_HOME=/node-shared/jerry007005/openpi_cache

cd /node-shared/jerry007005/haochuan/openpi
source .venv/bin/activate

echo "Node: $(hostname)"; nvidia-smi -L
cd /node-shared/jerry007005/haochuan/trossen_vla_bridge

python3 - <<'PY'
"""Run pi0 inference N times on TWO versions of the same training frame:
   - BEFORE-5cadbe2 path: image cv2-resized to 224x224 (what the client used to send)
   - AFTER-5cadbe2 path:  raw 480x640 (what the client sends now)
Report L1 err mean +/- std for each so we can compare past pi0's stochasticity.
"""
import sys, numpy as np
sys.path.insert(0, "src")
from trossen_vla_bridge.serve_pi0 import Pi0Policy
import pyarrow.parquet as pq, av, cv2
from pathlib import Path

ROOT = Path("/node-shared/jerry007005/haochuan/dataset/trossen_6task_combined")
EP, STEP = 10, 100

df = pq.read_table(ROOT / "data/chunk-000" / f"episode_{EP:06d}.parquet").to_pandas(zero_copy_only=False)
state = np.asarray(df["observation.state"].iloc[STEP], dtype=np.float32)
gt_action = np.asarray(df["action"].iloc[STEP], dtype=np.float32)
gt_delta = gt_action - state

def decode_frame(p, k):
    with av.open(str(p)) as c:
        s = c.streams.video[0]
        for i, f in enumerate(c.decode(s)):
            if i == k: return f.to_ndarray(format="rgb24")
    raise IndexError(k)

main_hwc  = decode_frame(ROOT / "videos/chunk-000/observation.images.cam_main"  / f"episode_{EP:06d}.mp4", STEP)
wrist_hwc = decode_frame(ROOT / "videos/chunk-000/observation.images.cam_wrist" / f"episode_{EP:06d}.mp4", STEP)
print(f"raw frames: main {main_hwc.shape}, wrist {wrist_hwc.shape}")

# BEFORE-5cadbe2: client did cv2.resize -> 224x224 -> CHW
main_224  = cv2.resize(main_hwc,  (224, 224)).transpose(2, 0, 1).copy()
wrist_224 = cv2.resize(wrist_hwc, (224, 224)).transpose(2, 0, 1).copy()
# AFTER-5cadbe2: raw 480x640 -> CHW
main_raw  = main_hwc.transpose(2, 0, 1).copy()
wrist_raw = wrist_hwc.transpose(2, 0, 1).copy()

print("loading pi0 policy from UoA-Trossen-Arm/pi0-trossen-6task ...")
policy = Pi0Policy(
    checkpoint="UoA-Trossen-Arm/pi0-trossen-6task",
    config_name="pi0_trossen_6task_low_mem_finetune",
)

def run(label, main_chw, wrist_chw, N=10):
    errs = []
    for i in range(N):
        out = policy.infer({
            "state": state,
            "images": {"cam_main": main_chw, "cam_wrist": wrist_chw},
            "prompt": "lift the eggplant",
        })
        delta = out["actions"][0] - state
        err = float(np.abs(delta - gt_delta).sum())
        errs.append(err)
    arr = np.array(errs)
    print(f"  {label:30s} N={N}  L1: mean={arr.mean():.4f}  std={arr.std():.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
    print(f"    individual: {[f'{x:.4f}' for x in errs]}")
    return arr

print(f"\nGT step-0 delta: {[f'{x:+.4f}' for x in gt_delta]}  L1={float(np.abs(gt_delta).sum()):.4f}\n")
print("=== BEFORE 5cadbe2 (cv2-resized 224x224) ===")
a_before = run("224x224 cv2", main_224, wrist_224, N=10)
print("\n=== AFTER 5cadbe2 (raw 480x640) ===")
a_after  = run("480x640 raw", main_raw, wrist_raw, N=10)

print(f"\n*** delta of means: before={a_before.mean():.4f}, after={a_after.mean():.4f}  "
      f"-> change={(a_after.mean() - a_before.mean()) / a_before.mean() * 100:+.1f}%")
PY
