"""openpi-client compatible websocket server wrapping OpenVLA (autoregressive).

OpenVLA predicts one 7-DoF action per forward pass. We expose it via the
openpi-client protocol by returning chunks of size 1 — the client re-queries
every step.

Run (use openvla's venv on a CUDA box):
    python serve_openvla.py \
        --checkpoint UoA-Trossen-Arm/openvla-7b-trossen-6task \
        --unnorm-key trossen_6task_combined --port 8000
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)

# Register OpenVLA's custom HF classes BEFORE any from_pretrained call.
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
AutoConfig.register("openvla", OpenVLAConfig)
AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

from openpi_client.base_policy import BasePolicy
from trossen_vla_bridge._websocket_policy_server import WebsocketPolicyServer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _resolve_checkpoint(path_or_repo: str) -> str:
    """If `path_or_repo` looks like a HF repo id, snapshot_download it."""
    if Path(path_or_repo).exists():
        return path_or_repo
    if "/" in path_or_repo and not path_or_repo.startswith("/"):
        log.info("Downloading %s from Hugging Face Hub", path_or_repo)
        return snapshot_download(repo_id=path_or_repo)
    raise FileNotFoundError(path_or_repo)


class OpenVLAPolicy(BasePolicy):
    def __init__(
        self,
        checkpoint: str,
        image_key: str = "cam_main",
        unnorm_key: str = "trossen_6task_combined",
        device: str = "cuda",
    ) -> None:
        ckpt = _resolve_checkpoint(checkpoint)
        log.info("Loading processor + model from %s", ckpt)
        self.processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
        self.vla = (
            AutoModelForVision2Seq.from_pretrained(
                ckpt,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            .to(device)
            .eval()
        )
        # The checkpoint's saved `dataset_statistics.json` contains our fine-tune
        # dataset's normalisation stats; merge it into the model's built-in
        # `norm_stats` so `predict_action(unnorm_key=...)` can find them.
        stats_path = os.path.join(ckpt, "dataset_statistics.json")
        if os.path.isfile(stats_path):
            with open(stats_path) as f:
                extra = json.load(f)
            if not hasattr(self.vla, "norm_stats") or self.vla.norm_stats is None:
                self.vla.norm_stats = {}
            self.vla.norm_stats.update(extra)
            log.info("Merged %d extra unnorm key(s) from %s: %s",
                     len(extra), stats_path, list(extra.keys()))
        else:
            log.warning("No dataset_statistics.json at %s — predict_action may fail",
                        ckpt)

        self.image_key = image_key
        self.unnorm_key = unnorm_key
        self.device = device
        if unnorm_key not in self.vla.norm_stats:
            raise KeyError(
                f"unnorm_key {unnorm_key!r} not found in model norm_stats; "
                f"available: {sorted(self.vla.norm_stats.keys())}"
            )
        log.info("OpenVLA ready (image_key=%s, unnorm_key=%s)", image_key, unnorm_key)

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        # Protocol:
        #   obs["images"][image_key]: uint8 CHW (3,H,W)
        #   obs["prompt"]: str
        #   (state is ignored — OpenVLA-7B uses no proprio.)
        if "images" not in obs or self.image_key not in obs["images"]:
            raise KeyError(
                f"obs['images'][{self.image_key!r}] is required; got keys: "
                f"{list(obs.get('images', {}).keys())}"
            )

        img_chw = np.asarray(obs["images"][self.image_key])
        if img_chw.ndim != 3 or img_chw.shape[0] != 3:
            raise ValueError(f"image must be uint8 CHW (3,H,W); got {img_chw.shape}")
        img_hwc = np.transpose(img_chw, (1, 2, 0)).astype(np.uint8)
        # Pre-resize to 224x224 with cv2.INTER_LINEAR -- training ran every
        # frame through OXE's tf.image.resize(224, 224) before the model saw
        # it, and cv2.INTER_LINEAR is the inference-side resize that most
        # closely matches tf.image.resize at the pixel level (letting
        # PrismaticImageProcessor's TVF.resize do it instead empirically
        # regressed L1 by ~28% on training-distribution frames).
        if img_hwc.shape[:2] != (224, 224):
            import cv2
            img_hwc = cv2.resize(img_hwc, (224, 224), interpolation=cv2.INTER_LINEAR)
        image = Image.fromarray(img_hwc)

        prompt_text = obs.get("prompt") or "lift the eggplant"
        prompt = f"In: What action should the robot take to {prompt_text.lower()}?\nOut:"
        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
        action = self.vla.predict_action(
            **inputs, unnorm_key=self.unnorm_key, do_sample=False
        )
        action = np.asarray(action, dtype=np.float32)  # (7,)

        # Add state back to the arm dims to reconstruct an absolute joint target.
        # The forge_dataset_transform subtracts state from action[:, :6] during
        # training, so the model output here is a per-step delta for arm joints
        # (gripper dim 6 stays absolute). pi0 does the equivalent internally via
        # its AbsoluteActions output transform.
        state = obs.get("state")
        if state is None:
            raise KeyError("obs['state'] is required (model is delta-trained)")
        state = np.asarray(state, dtype=np.float32)
        if state.shape[0] < 6:
            raise ValueError(f"state must have >=6 dims; got {state.shape}")
        action[:6] = action[:6] + state[:6]

        return {"actions": action.reshape(1, -1)}  # chunk size = 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="UoA-Trossen-Arm/openvla-7b-trossen-6task",
        help="HF repo id or local checkpoint dir",
    )
    parser.add_argument("--image-key", default="cam_main")
    parser.add_argument("--unnorm-key", default="trossen_6task_combined")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    policy = OpenVLAPolicy(
        checkpoint=args.checkpoint,
        image_key=args.image_key,
        unnorm_key=args.unnorm_key,
        device=args.device,
    )
    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={
            "vla": "openvla-7b",
            "action_dim": 7,
            "chunk_size": 1,
            "expected_image_keys": [args.image_key],
        },
    )
    log.info("Listening on %s:%d", args.host, args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
