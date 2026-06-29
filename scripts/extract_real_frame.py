"""Pull a single real (state, cam_main, cam_wrist) sample from the merged
trossen_6task_combined LeRobot dataset, and save it as .npy / .npz for the
3 model-compare smoke tests.

Frame chosen: episode 10 (eggplant task), step 100 -- mid-trajectory.
"""
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
import av

ROOT = Path("/node-shared/jerry007005/haochuan/dataset/trossen_6task_combined")
OUT  = Path("/node-shared/jerry007005/haochuan/trossen_vla_bridge/scripts/_real_frame.npz")

EP = 10
STEP = 100

# state from parquet
pq_path = ROOT / "data" / "chunk-000" / f"episode_{EP:06d}.parquet"
df = pq.read_table(pq_path).to_pandas(zero_copy_only=False)
state = np.asarray(df["observation.state"].iloc[STEP], dtype=np.float32)
action = np.asarray(df["action"].iloc[STEP], dtype=np.float32)
print(f"ep {EP} step {STEP}: state={state.tolist()}")
print(f"                  action (gt)={action.tolist()}")

# Decode the matching frame from both cameras' mp4 with pyav.
def decode_frame(mp4_path: Path, step: int) -> np.ndarray:
    with av.open(str(mp4_path)) as container:
        stream = container.streams.video[0]
        # Just iterate; the file is short.
        for i, frame in enumerate(container.decode(stream)):
            if i == step:
                img = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
                return img
    raise IndexError(f"step {step} beyond {mp4_path}")

img_main_hwc  = decode_frame(ROOT / "videos" / "chunk-000" / "observation.images.cam_main"  / f"episode_{EP:06d}.mp4", STEP)
img_wrist_hwc = decode_frame(ROOT / "videos" / "chunk-000" / "observation.images.cam_wrist" / f"episode_{EP:06d}.mp4", STEP)
print(f"cam_main shape  = {img_main_hwc.shape}, dtype={img_main_hwc.dtype}, mean={img_main_hwc.mean():.1f}")
print(f"cam_wrist shape = {img_wrist_hwc.shape}, dtype={img_wrist_hwc.dtype}, mean={img_wrist_hwc.mean():.1f}")

# Keep the raw 480x640 frame -- the bridge client now forwards full-res to
# the server so the model's own processor can apply its training-time
# letterbox + resize pipeline. Just transpose to CHW uint8 for the protocol.
main_chw  = img_main_hwc.transpose(2, 0, 1).copy()
wrist_chw = img_wrist_hwc.transpose(2, 0, 1).copy()

np.savez(OUT,
         state=state,
         action_gt=action,
         cam_main=main_chw,
         cam_wrist=wrist_chw,
         prompt=np.array("lift the eggplant"))
print(f"saved {OUT} (state shape={state.shape}, cam_main shape={main_chw.shape}, cam_wrist shape={wrist_chw.shape})")
