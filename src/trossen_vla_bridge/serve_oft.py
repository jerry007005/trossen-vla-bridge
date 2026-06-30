"""openpi-client compatible websocket server wrapping OpenVLA-OFT.

OFT predicts a chunk of NUM_ACTIONS_CHUNK (=20 for Trossen) 7-DoF actions per
forward pass through the LoRA-merged base + a separate L1 regression head.

Run (use openvla-oft's venv on a CUDA box):
    python serve_oft.py \
        --checkpoint UoA-Trossen-Arm/openvla-7b-oft-trossen-6task \
        --unnorm-key trossen_6task_combined --port 8000
"""
from __future__ import annotations

# IMPORTANT: openvla-oft picks robot-platform constants at *import time* by
# scanning sys.argv for "libero"/"aloha"/"bridge"/"trossen". We force "trossen"
# in argv BEFORE importing prismatic.vla.constants so TROSSEN_CONSTANTS
# (chunk=20, action_dim=7) is selected even when this script is launched without
# a `--run_id_note trossen-...` arg.
import sys
if not any("trossen" in arg.lower() for arg in sys.argv):
    sys.argv.append("--robot=trossen")  # synthetic, just to pick TROSSEN_CONSTANTS

import argparse
import logging
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from dataclasses import dataclass

from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

# We avoid importing the official GenerateConfig because it lives under
# `experiments.robot.libero.run_libero_eval`, which transitively imports the
# `libero` benchmark package — that's not installed in this venv. The
# `openvla_utils.get_*` helpers only access the handful of fields below.
@dataclass
class GenerateConfig:
    pretrained_checkpoint: str
    use_l1_regression: bool = True
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    use_film: bool = False
    num_images_in_input: int = 2
    use_proprio: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True
    unnorm_key: str = "forge_dataset"

from experiments.robot.openvla_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)

from openpi_client.base_policy import BasePolicy
from trossen_vla_bridge._websocket_policy_server import WebsocketPolicyServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _resolve_checkpoint(path_or_repo: str) -> str:
    if Path(path_or_repo).exists():
        return path_or_repo
    if "/" in path_or_repo and not path_or_repo.startswith("/"):
        log.info("Downloading %s from Hugging Face Hub", path_or_repo)
        return snapshot_download(repo_id=path_or_repo)
    raise FileNotFoundError(path_or_repo)


class OpenVLAOFTPolicy(BasePolicy):
    def __init__(
        self,
        checkpoint: str,
        unnorm_key: str = "forge_dataset",
        primary_image_key: str = "cam_main",
        wrist_image_key: str = "cam_wrist",
    ) -> None:
        ckpt = _resolve_checkpoint(checkpoint)
        self.cfg = GenerateConfig(
            pretrained_checkpoint=ckpt,
            use_l1_regression=True,
            use_diffusion=False,
            use_film=False,
            num_images_in_input=2,
            use_proprio=True,  # 6-task ckpt was TRAINED with use_proprio=True;
                               # leaving this False silently disabled proprio at
                               # inference and the model's L1 head fell back to
                               # near-mean actions (visible as a stalled arm).
            load_in_8bit=False,
            load_in_4bit=False,
            center_crop=True,
            unnorm_key=unnorm_key,
        )
        log.info("Loading OFT VLA + L1 action head + proprio projector from %s", ckpt)
        self.vla = get_vla(self.cfg)
        self.action_head = get_action_head(self.cfg, llm_dim=self.vla.llm_dim)
        self.proprio_projector = get_proprio_projector(
            self.cfg, llm_dim=self.vla.llm_dim, proprio_dim=ACTION_DIM
        )
        self.processor = get_processor(self.cfg)
        self.primary_image_key = primary_image_key
        self.wrist_image_key = wrist_image_key
        log.info(
            "OFT ready: chunk=%d action_dim=%d primary=%s wrist=%s",
            NUM_ACTIONS_CHUNK,
            ACTION_DIM,
            primary_image_key,
            wrist_image_key,
        )

    def infer(self, obs: dict) -> dict:
        # Protocol:
        #   obs["images"][primary_image_key]: uint8 CHW
        #   obs["images"][wrist_image_key]:   uint8 CHW
        #   obs["prompt"]: str
        imgs = obs.get("images") or {}
        if self.primary_image_key not in imgs:
            raise KeyError(
                f"obs['images'][{self.primary_image_key!r}] is required; "
                f"got {list(imgs.keys())}"
            )
        if self.wrist_image_key not in imgs:
            raise KeyError(
                f"obs['images'][{self.wrist_image_key!r}] is required; "
                f"got {list(imgs.keys())}"
            )

        def chw_to_hwc224(x: np.ndarray) -> np.ndarray:
            """CHW uint8 (any HxW) -> HWC uint8 224x224 via cv2.INTER_LINEAR.

            Training ran every frame through OXE's tf.image.resize(224, 224)
            before the model saw it (see prismatic/vla/datasets/rlds/
            obs_transforms.decode_and_resize). cv2.INTER_LINEAR is the
            inference-side resize whose pixel-level output most closely
            tracks tf.image.resize -- letting the PrismaticImageProcessor
            fall back to torchvision's TVF.resize empirically regressed L1.
            """
            import cv2  # cheap; only imported when this code path runs
            arr = np.asarray(x)
            if arr.ndim != 3 or arr.shape[0] != 3:
                raise ValueError(
                    f"image must be uint8 CHW (3,H,W); got shape {arr.shape}"
                )
            hwc = np.transpose(arr, (1, 2, 0)).astype(np.uint8)
            if hwc.shape[:2] != (224, 224):
                hwc = cv2.resize(hwc, (224, 224), interpolation=cv2.INTER_LINEAR)
            return hwc

        # get_vla_action consumes:
        #   oft_obs["full_image"]  HWC uint8 primary camera
        #   oft_obs["wrist_image"] HWC uint8 wrist camera
        #   oft_obs["state"]       7-D proprio (raw radian state); normalised
        #                          inside get_vla_action via norm_stats[unnorm_key]["proprio"]
        state_raw = obs.get("state")
        if state_raw is None:
            raise KeyError("obs['state'] is required (model is delta-trained + uses proprio)")
        state_raw = np.asarray(state_raw, dtype=np.float32)
        if state_raw.shape[0] < ACTION_DIM:
            raise ValueError(f"state must have >= {ACTION_DIM} dims; got {state_raw.shape}")
        oft_obs = {
            "full_image": chw_to_hwc224(imgs[self.primary_image_key]),
            "wrist_image": chw_to_hwc224(imgs[self.wrist_image_key]),
            "state": state_raw,
        }
        task = obs.get("prompt") or "lift the eggplant"
        actions = get_vla_action(
            self.cfg,
            self.vla,
            self.processor,
            oft_obs,
            task_label=task,
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
        )
        # `actions` is a list of NUM_ACTIONS_CHUNK numpy arrays, each (ACTION_DIM,)
        actions = np.stack(actions, axis=0).astype(np.float32)

        # Add state back to every step's arm dims to reconstruct absolute joint
        # targets. forge_dataset_transform subtracted state from action[:, :6]
        # during training, so the model output here is a per-step arm delta.
        state = obs.get("state")
        if state is None:
            raise KeyError("obs['state'] is required (model is delta-trained)")
        state = np.asarray(state, dtype=np.float32)
        if state.shape[0] < 6:
            raise ValueError(f"state must have >=6 dims; got {state.shape}")
        actions[:, :6] = actions[:, :6] + state[None, :6]

        return {"actions": actions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="UoA-Trossen-Arm/openvla-7b-oft-trossen-6task",
        help="HF repo id or local checkpoint dir",
    )
    parser.add_argument("--primary-image-key", default="cam_main")
    parser.add_argument("--wrist-image-key", default="cam_wrist")
    parser.add_argument("--unnorm-key", default="trossen_6task_combined")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    policy = OpenVLAOFTPolicy(
        checkpoint=args.checkpoint,
        unnorm_key=args.unnorm_key,
        primary_image_key=args.primary_image_key,
        wrist_image_key=args.wrist_image_key,
    )
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={
            "vla": "openvla-7b-oft",
            "action_dim": int(ACTION_DIM),
            "chunk_size": int(NUM_ACTIONS_CHUNK),
            "expected_image_keys": [args.primary_image_key, args.wrist_image_key],
        },
    )
    log.info("Listening on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
