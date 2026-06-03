"""openpi-client compatible websocket server wrapping our pi0 (PyTorch) checkpoint.

The pi0 model returns a chunk of 50 × 7 action steps per forward pass.

Run (use openpi's venv):
    python serve_pi0.py \
        --checkpoint UoA-Trossen-Arm/pi0-lift-eggplant-pytorch \
        --config-name pi0_trossen_lift_eggplant_low_mem_finetune \
        --port 8000

NOTE: this venv pins torch==2.7.1, which lacks Blackwell (sm_120) kernels — on a
Blackwell node the model will run on CPU (still produces correct actions, just
slow). On H100/H200/A100 it runs on GPU.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from openpi.training import config as _config
from openpi.policies import policy_config

from openpi_client.base_policy import BasePolicy
from _websocket_policy_server import WebsocketPolicyServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _resolve_checkpoint(path_or_repo: str) -> str:
    if Path(path_or_repo).exists():
        return path_or_repo
    if "/" in path_or_repo and not path_or_repo.startswith("/"):
        log.info("Downloading %s from Hugging Face Hub", path_or_repo)
        return snapshot_download(repo_id=path_or_repo)
    raise FileNotFoundError(path_or_repo)


class Pi0Policy(BasePolicy):
    def __init__(
        self,
        checkpoint: str,
        config_name: str,
        primary_image_key: str = "cam_main",
        wrist_image_key: str = "cam_wrist",
    ) -> None:
        ckpt = _resolve_checkpoint(checkpoint)
        log.info("Resolving openpi config %s", config_name)
        cfg = _config.get_config(config_name)
        log.info("Creating pi0 policy from %s", ckpt)
        self.policy = policy_config.create_trained_policy(cfg, ckpt)
        self.primary_image_key = primary_image_key
        self.wrist_image_key = wrist_image_key
        log.info("pi0 ready: chunk=%d action_dim=%d",
                 cfg.model.action_horizon, cfg.model.action_dim)

        # Warm up: trigger torch.compile / autotune now (can take 5-10 min for
        # max-autotune mode) so the first real client call doesn't block the
        # event loop and trip a websocket handshake/keepalive timeout.
        log.info("Warming up pi0 with dummy obs (this may take a few minutes)...")
        try:
            dummy_state = np.zeros(7, dtype=np.float32)
            dummy_img = np.zeros((3, 224, 224), dtype=np.uint8)
            self.infer({
                "state": dummy_state,
                "images": {primary_image_key: dummy_img, wrist_image_key: dummy_img},
                "prompt": "lift the eggplant",
            })
            log.info("Warm-up complete.")
        except Exception as exc:
            log.warning("Warm-up infer raised: %s (continuing anyway)", exc)

    def infer(self, obs: dict) -> dict:
        # openpi-client protocol obs has CHW uint8 images. The TrossenInputs
        # transform inside the policy expects HWC uint8 under nested keys.
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

        def chw_to_hwc(x):
            arr = np.asarray(x)
            if arr.ndim != 3 or arr.shape[0] != 3:
                raise ValueError(f"image must be uint8 CHW (3,H,W); got {arr.shape}")
            return np.transpose(arr, (1, 2, 0)).astype(np.uint8)

        state = np.asarray(obs.get("state"), dtype=np.float32)
        if state.shape != (7,):
            raise ValueError(f"state must be float32 (7,); got {state.shape}")

        pi0_obs = {
            "observation/image":        chw_to_hwc(imgs[self.primary_image_key]),
            "observation/wrist_image":  chw_to_hwc(imgs[self.wrist_image_key]),
            "observation/state":        state,
            "prompt":                   obs.get("prompt") or "lift the eggplant",
        }
        out = self.policy.infer(pi0_obs)
        # out["actions"] shape (action_horizon, action_dim) = (50, 7) for our cfg.
        actions = np.asarray(out["actions"], dtype=np.float32)
        return {"actions": actions}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="UoA-Trossen-Arm/pi0-lift-eggplant-pytorch")
    p.add_argument("--config-name", default="pi0_trossen_lift_eggplant_low_mem_finetune")
    p.add_argument("--primary-image-key", default="cam_main")
    p.add_argument("--wrist-image-key", default="cam_wrist")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    # We need HF_LEROBOT_HOME for the get_config call to find norm stats when
    # the dataset transforms are materialised.
    os.environ.setdefault("HF_LEROBOT_HOME", "/node-shared/jerry007005/haochuan/lerobot_home")

    policy = Pi0Policy(
        checkpoint=args.checkpoint,
        config_name=args.config_name,
        primary_image_key=args.primary_image_key,
        wrist_image_key=args.wrist_image_key,
    )
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={
            "vla": "pi0",
            "action_dim": 7,
            "chunk_size": 50,
            "expected_image_keys": [args.primary_image_key, args.wrist_image_key],
        },
    )
    log.info("Listening on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
