"""openpi-client compatible websocket server wrapping OpenVLA-OFT.

OFT predicts a chunk of NUM_ACTIONS_CHUNK (=20 for Trossen) 7-DoF actions per
forward pass through the LoRA-merged base + either an L1 regression head or a
DDIM diffusion action head.

Run (use openvla-oft's venv on a CUDA box):

    # Default: diverse-pick-and-place L1 head
    python serve_oft.py --port 8000

    # 6-task multi-task L1 head
    python serve_oft.py \
        --checkpoint UoA-Trossen-Arm/openvla-7b-oft-trossen-6task-png \
        --unnorm-key trossen_6task_combined_png --port 8000

    # 6-task diffusion head (auto-detected from checkpoint name)
    python serve_oft.py \
        --checkpoint UoA-Trossen-Arm/openvla-7b-oft-trossen-6task-png-diffusion \
        --unnorm-key trossen_6task_combined_png --port 8000
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
    get_noisy_action_projector,
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
        unnorm_key: str = "diverse_pnp_png",
        primary_image_key: str = "cam_main",
        wrist_image_key: str = "cam_wrist",
        use_diffusion: bool = False,
    ) -> None:
        ckpt = _resolve_checkpoint(checkpoint)
        self.use_diffusion = use_diffusion
        self.cfg = GenerateConfig(
            pretrained_checkpoint=ckpt,
            use_l1_regression=not use_diffusion,
            use_diffusion=use_diffusion,
            use_film=False,
            num_images_in_input=2,
            use_proprio=True,
            load_in_8bit=False,
            load_in_4bit=False,
            center_crop=True,
            unnorm_key=unnorm_key,
        )
        head_name = "diffusion" if use_diffusion else "L1"
        log.info("Loading OFT VLA + %s action head + proprio projector from %s", head_name, ckpt)
        self.vla = get_vla(self.cfg)
        self.action_head = get_action_head(self.cfg, llm_dim=self.vla.llm_dim)
        self.proprio_projector = get_proprio_projector(
            self.cfg, llm_dim=self.vla.llm_dim, proprio_dim=ACTION_DIM
        )
        # Diffusion head also needs the noisy-action projector at inference
        # (it feeds noised action samples back into the LLM at each denoising step).
        self.noisy_action_projector = (
            get_noisy_action_projector(self.cfg, llm_dim=self.vla.llm_dim)
            if use_diffusion
            else None
        )
        self.processor = get_processor(self.cfg)
        self.primary_image_key = primary_image_key
        self.wrist_image_key = wrist_image_key
        log.info(
            "OFT ready: head=%s chunk=%d action_dim=%d primary=%s wrist=%s",
            head_name,
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

        def chw_to_hwc(x: np.ndarray) -> np.ndarray:
            """CHW uint8 (any HxW) -> HWC uint8 (any HxW).

            The PrismaticImageProcessor (via TVF.resize) handles resizing to
            (224, 224) internally -- this matches training, where the new
            *_png datasets disable OXE's tf.image.resize so the model's own
            processor is the only resize step in the pipeline.
            """
            arr = np.asarray(x)
            if arr.ndim != 3 or arr.shape[0] != 3:
                raise ValueError(
                    f"image must be uint8 CHW (3,H,W); got shape {arr.shape}"
                )
            return np.transpose(arr, (1, 2, 0)).astype(np.uint8)

        state_raw = obs.get("state")
        if state_raw is None:
            raise KeyError("obs['state'] is required (proprio projector input)")
        state_raw = np.asarray(state_raw, dtype=np.float32)
        if state_raw.shape[0] < ACTION_DIM:
            raise ValueError(f"state must have >= {ACTION_DIM} dims; got {state_raw.shape}")
        oft_obs = {
            "full_image": chw_to_hwc(imgs[self.primary_image_key]),
            "wrist_image": chw_to_hwc(imgs[self.wrist_image_key]),
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
            noisy_action_projector=self.noisy_action_projector,
        )
        # `actions` is a list of NUM_ACTIONS_CHUNK numpy arrays, each (ACTION_DIM,).
        # The model is trained ALOHA-style on absolute joint targets (no per-step
        # delta), so we DO NOT add state back -- whatever the model returns is
        # already the absolute Goal_Position the robot should reach.
        actions = np.stack(actions, axis=0).astype(np.float32)

        return {"actions": actions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="UoA-Trossen-Arm/openvla-7b-oft-diverse-pnp-png",
        help="HF repo id or local checkpoint dir",
    )
    parser.add_argument("--primary-image-key", default="cam_main")
    parser.add_argument("--wrist-image-key", default="cam_wrist")
    parser.add_argument("--unnorm-key", default="diverse_pnp_png")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--use-diffusion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the DDIM diffusion action head. If unset, auto-detected from "
             "checkpoint name (any '-diffusion' substring turns it on).",
    )
    args = parser.parse_args()

    if args.use_diffusion is None:
        args.use_diffusion = "diffusion" in args.checkpoint.lower()
        if args.use_diffusion:
            log.info("Auto-detected diffusion head from checkpoint name")

    policy = OpenVLAOFTPolicy(
        checkpoint=args.checkpoint,
        unnorm_key=args.unnorm_key,
        primary_image_key=args.primary_image_key,
        wrist_image_key=args.wrist_image_key,
        use_diffusion=args.use_diffusion,
    )
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={
            "vla": "openvla-7b-oft",
            "head": "diffusion" if args.use_diffusion else "l1",
            "action_dim": int(ACTION_DIM),
            "chunk_size": int(NUM_ACTIONS_CHUNK),
            "expected_image_keys": [args.primary_image_key, args.wrist_image_key],
        },
    )
    log.info("Listening on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
