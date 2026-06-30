"""Trossen AI Solo (single arm, 7-DoF) <-> openpi-client policy server bridge.

Adapted from TrossenRobotics/openpi/examples/trossen_ai/main.py (bimanual).

Reads (state, cam_main, cam_wrist) from the arm via the existing lerobot
trossen-ai fork (`lerobot.common.robot_devices.*`), forwards to a policy
server, and replays the returned action chunk on the follower arm.

Run (in /node-shared/jerry007005/haochuan/lerobot/.venv):
    # No --task-prompt: an interactive picker prompts you to choose one of the
    # 6 trained task strings (or type a custom one) at startup.
    python client_trossen_solo.py --mode test --server-host localhost --server-port 8000

    # Specify the task directly:
    python client_trossen_solo.py \
        --mode test --task-prompt "pick up the carrot and place it in the plate" \
        --server-host localhost --server-port 8000

    # Real motion (DO NOT RUN until the model is verified safe):
    python client_trossen_solo.py --mode autonomous \
        --task-prompt "lift the pineapple"
"""
from __future__ import annotations

import argparse
import logging
import time
from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy

# Old-fork lerobot API (Interbotix v0.1.0). When you move to the lerobot_trossen
# plugin (v0.4+) swap these for `from lerobot.robots import make_robot_from_config`
# and rebuild the config object — the rest of this file should be untouched.
from lerobot.common.robot_devices.robots.configs import TrossenAISoloRobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ACTION_DIM = 7  # 6 arm joints + 1 gripper (matches TROSSEN_CONSTANTS)

# Empirically constant start pose for every episode in trossen_6task_combined
# (std < 1e-3 rad across all 1200 episodes / 6 tasks). Without resetting to it
# at the top of each episode the model gets out-of-distribution state on
# episode 2+ and stalls. Values:
#   j0 ~ 0       j1 = pi/3 (60°)   j2 = pi/6 (30°)   j3 ~ 0.6284  (36°)
#   j4 ~ 0       j5 ~ 0            gripper = 0 (open)
TROSSEN_6TASK_HOME_POSE = np.array(
    [0.0, 1.0470, 0.5233, 0.6284, 0.0, 0.0, 0.0],
    dtype=np.float32,
)


class TrossenSoloPolicyBridge:
    """Single-arm Trossen AI Solo bridge to an openpi-client policy server."""

    def __init__(
        self,
        server_host: str = "localhost",
        server_port: int = 8000,
        control_frequency: int = 20,
        rate_of_inference: int = 1,
        max_steps: int = 1000,
        mode: str = "test",
        camera_interface: str = "intel_realsense",
        send_wrist: bool = True,
        follower_ip: str | None = None,
        leader_ip: str | None = None,
        no_leader: bool = True,
        cam_main_serial: str | None = None,
        cam_wrist_serial: str | None = None,
        reset_to_home: bool = False,
    ) -> None:
        if mode not in {"autonomous", "test"}:
            raise ValueError(f"mode must be 'autonomous' or 'test'; got {mode!r}")
        self.mode = mode
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.rate_of_inference = rate_of_inference
        self.max_steps = max_steps
        self.send_wrist = send_wrist
        self.reset_to_home = reset_to_home

        log.info("Connecting to policy server %s:%d", server_host, server_port)
        self.policy = WebsocketClientPolicy(host=server_host, port=server_port)
        meta = self.policy.get_server_metadata()
        log.info("Server metadata: %s", meta)
        self.action_chunk_size = int(meta.get("chunk_size", 1))
        if self.rate_of_inference > self.action_chunk_size:
            log.warning(
                "rate_of_inference=%d > server chunk_size=%d; capping",
                self.rate_of_inference, self.action_chunk_size,
            )
            self.rate_of_inference = self.action_chunk_size

        log.info("Initialising Trossen AI Solo robot (camera=%s)", camera_interface)
        robot_cfg = TrossenAISoloRobotConfig(camera_interface=camera_interface)
        # Apply hardware overrides BEFORE the robot is constructed. Defaults in
        # the upstream config are: follower 192.168.1.3, leader 192.168.1.2,
        # cam_main RealSense SN 130322270184, cam_wrist SN 218622274938.
        if follower_ip is not None:
            robot_cfg.follower_arms["main"].ip = follower_ip
            log.info("  follower IP → %s", follower_ip)
        if no_leader:
            # Autonomous policy execution does not need a leader arm — drop it
            # so robot.connect() does not try to reach 192.168.1.2.
            robot_cfg.leader_arms = {}
            log.info("  leader arm dropped (autonomous-only mode)")
        elif leader_ip is not None:
            robot_cfg.leader_arms["main"].ip = leader_ip
            log.info("  leader IP → %s", leader_ip)
        if cam_main_serial is not None and "cam_main" in robot_cfg.cameras:
            robot_cfg.cameras["cam_main"].serial_number = int(cam_main_serial)
            log.info("  cam_main serial → %s", cam_main_serial)
        if cam_wrist_serial is not None and "cam_wrist" in robot_cfg.cameras:
            robot_cfg.cameras["cam_wrist"].serial_number = int(cam_wrist_serial)
            log.info("  cam_wrist serial → %s", cam_wrist_serial)

        self.robot = make_robot_from_config(robot_cfg)
        self.robot.connect()

        self.episode_step = 0
        self.current_chunk: np.ndarray | None = None
        self.chunk_idx = 0
        # buffer of (step -> list of candidate actions), used for temporal
        # ensembling if you later switch it on.
        self.action_buffer: Dict[int, List[np.ndarray]] = defaultdict(list)

    # ------------------------------------------------------------------ obs

    def _read_obs(self) -> dict:
        observation_dict = self.robot.capture_observation()

        # 1. State: 7 follower joint positions
        joint_keys = sorted(
            k for k in observation_dict
            if k.startswith("observation.state") or k.endswith(".pos")
        )
        joint_vals: list[float] = []
        for k in joint_keys:
            v = observation_dict[k]
            try:
                joint_vals.extend(np.asarray(v, dtype=np.float32).reshape(-1).tolist())
            except Exception:
                pass
        if len(joint_vals) < ACTION_DIM:
            raise RuntimeError(
                f"Expected at least {ACTION_DIM} joint positions, got {len(joint_vals)} "
                f"from keys {joint_keys}"
            )
        state = np.asarray(joint_vals[:ACTION_DIM], dtype=np.float32)

        # 2. Images: lerobot returns HWC RGB. Send RAW 480x640 CHW to the
        # server -- the OFT and openvla servers will cv2-resize to 224x224
        # internally to match the tf.image.resize pixel-level behaviour their
        # training data went through. The pi0 server keeps the raw frame as-is
        # and lets openpi's image pipeline handle it.
        images: dict[str, np.ndarray] = {}
        wanted = ["cam_main"] + (["cam_wrist"] if self.send_wrist else [])
        for cam in wanted:
            key = f"observation.images.{cam}"
            if key not in observation_dict:
                # fall back to plain key (different lerobot versions name differently)
                alt = next((k for k in observation_dict if cam in k and "images" in k), None)
                if alt is None:
                    raise KeyError(f"camera '{cam}' not found in obs keys {list(observation_dict)}")
                key = alt
            img = np.asarray(observation_dict[key])
            if img.dtype != np.uint8:
                # lerobot often gives float in [0,1]; convert.
                img = (img * 255.0).clip(0, 255).astype(np.uint8)
            if img.ndim == 3 and img.shape[0] in (1, 3):
                # CHW already
                images[cam] = img
            else:
                # HWC -> CHW
                images[cam] = np.transpose(img, (2, 0, 1))

        return {
            "state":  state,
            "images": images,
            "prompt": getattr(self, "_task_prompt", "lift the eggplant"),
        }

    # ------------------------------------------------------------------ act

    def _send_action(self, action: np.ndarray) -> None:
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"action shape must be ({ACTION_DIM},); got {action.shape}")
        if self.mode == "test":
            log.debug("test-mode action (not sent): %s", np.round(action, 4))
            return
        # The follower arm expects an action under the key
        # `action.{joint_name}` per joint; lerobot 0.1.0's ManipulatorRobot
        # consumes a torch.tensor of shape (ACTION_DIM,).
        import torch
        self.robot.send_action(torch.from_numpy(action.astype(np.float32)))

    # ------------------------------------------------------------- run loop

    def _move_to_home(self, n_steps: int = 60) -> None:
        """Linearly interpolate from the current pose to TROSSEN_6TASK_HOME_POSE
        over `n_steps` (= 3 s at 20 Hz)."""
        if self.mode == "test":
            log.info("[home] skipping reset in test mode")
            return
        cur = self._read_obs()["state"]
        target = TROSSEN_6TASK_HOME_POSE
        log.info("[home] resetting from %s to %s over %d steps",
                 np.round(cur, 4).tolist(), target.tolist(), n_steps)
        for k in range(1, n_steps + 1):
            tick = time.perf_counter()
            alpha = k / n_steps
            mid = (1.0 - alpha) * cur + alpha * target
            self._send_action(mid.astype(np.float32))
            elapsed = time.perf_counter() - tick
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)
        log.info("[home] reset complete")

    def run_episode(self, task_prompt: str) -> None:
        self._task_prompt = task_prompt
        self.episode_step = 0
        self.current_chunk = None
        self.chunk_idx = 0
        log.info("Starting episode (mode=%s, prompt=%r, max_steps=%d)",
                 self.mode, task_prompt, self.max_steps)
        if self.reset_to_home:
            self._move_to_home()

        while self.episode_step < self.max_steps:
            tick = time.perf_counter()

            need_new_chunk = (
                self.current_chunk is None
                or self.chunk_idx >= self.rate_of_inference
            )
            if need_new_chunk:
                obs = self._read_obs()
                resp = self.policy.infer(obs)
                self.current_chunk = np.asarray(resp["actions"], dtype=np.float32)
                self.chunk_idx = 0
                log.info(
                    "step=%d  query  chunk=%s  server_timing=%s",
                    self.episode_step,
                    tuple(self.current_chunk.shape),
                    resp.get("server_timing"),
                )

            action = self.current_chunk[self.chunk_idx]
            self._send_action(action)
            self.chunk_idx += 1
            self.episode_step += 1

            elapsed = time.perf_counter() - tick
            if elapsed < self.dt:
                time.sleep(self.dt - elapsed)

        log.info("Episode finished at step %d", self.episode_step)

    def shutdown(self) -> None:
        try:
            self.robot.disconnect()
        except Exception as exc:
            log.warning("Robot disconnect raised: %s", exc)


# The six task strings that the 6-task pi0 / OFT models were trained on.
# Order matches merge_6task_lerobot.py task_index assignment.
TROSSEN_6TASK_PROMPTS = (
    "lift the eggplant",
    "lift the carrot",
    "pick up the carrot and place it in the plate",
    "lift the pineapple",
    "pick up the eggplant and place it in the plate",
    "pick up the pineapple and place it in the plate",
)


def _pick_task_prompt_interactively() -> str:
    """Show the six known task prompts and let the user pick one (or type a custom string)."""
    print("\nPick a task instruction (the trained models were finetuned on these 6):")
    for i, p in enumerate(TROSSEN_6TASK_PROMPTS, start=1):
        print(f"  [{i}] {p}")
    print("  [c] (custom) — type your own")
    while True:
        choice = input("Enter 1-6 or c: ").strip().lower()
        if choice in {str(i) for i in range(1, len(TROSSEN_6TASK_PROMPTS) + 1)}:
            prompt = TROSSEN_6TASK_PROMPTS[int(choice) - 1]
            print(f"Using: {prompt!r}")
            return prompt
        if choice == "c":
            custom = input("Custom prompt: ").strip()
            if custom:
                return custom
            print("Empty input, try again.")
            continue
        print(f"Invalid choice {choice!r}, try again.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["autonomous", "test"], default="test",
                        help="`test` reads obs + queries server but does NOT move the arm")
    parser.add_argument("--task-prompt", default=None,
                        help="Task instruction sent with every observation. If omitted, "
                             "an interactive picker is shown at startup with the 6 "
                             "Trossen 6-task prompts.")
    parser.add_argument("--server-host", default="localhost")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--control-frequency", type=int, default=20,
                        help="Hz at which the client sends actions. Must match the "
                             "training data fps (trossen_6task_combined was collected "
                             "at 20 Hz, so each step in the policy's action chunk is "
                             "spaced 50 ms apart; running at 30 Hz replays the chunk "
                             "1.5x too fast and the arm visibly stalls mid-task).")
    parser.add_argument("--rate-of-inference", type=int, default=1,
                        help="How many actions from each chunk to apply before requerying. "
                             "Default 1 (use only chunk[0] then ask again). This is required "
                             "for the OFT and openvla checkpoints we ship: their training "
                             "transform computes per-step delta = action[t+k] - state[t+k], "
                             "but at inference we only know the CURRENT state, so adding it "
                             "to every chunk step accumulates an error proportional to k. "
                             "Only chunk[0] is unbiased. pi0 uses the openpi convention "
                             "(delta relative to chunk-start state), so re-using a longer "
                             "chunk is safe -- bump this to e.g. 20 or 50 in that case to "
                             "get the chunked-policy throughput benefit.")
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--camera-interface", default="intel_realsense",
                        choices=["intel_realsense", "opencv"])
    parser.add_argument("--no-wrist", action="store_true",
                        help="Send only cam_main (use when serving plain OpenVLA)")
    parser.add_argument("--follower-ip", default=None,
                        help="Override follower arm IP (default config: 192.168.1.3)")
    parser.add_argument("--leader-ip", default=None,
                        help="Override leader arm IP (default config: 192.168.1.2). "
                             "Ignored when --no-leader is set.")
    parser.add_argument("--with-leader", action="store_true",
                        help="Connect to the leader arm too. Default is autonomous-only "
                             "(no leader required).")
    parser.add_argument("--cam-main-serial", default=None,
                        help="Override RealSense SN for cam_main (default 130322270184)")
    parser.add_argument("--cam-wrist-serial", default=None,
                        help="Override RealSense SN for cam_wrist (default 218622274938)")
    parser.add_argument("--connect-only", action="store_true",
                        help="Connect to robot, print one observation, disconnect. "
                             "Does NOT touch the policy server or motors.")
    parser.add_argument("--reset-to-home", action="store_true",
                        help="Before the policy loop starts, linearly interpolate from "
                             "the current pose to the trained home pose "
                             "(j1=pi/3, j2=pi/6, j3=0.6284, rest=0, gripper open) over "
                             "3 seconds. Off by default -- enable when starting a fresh "
                             "episode and the arm is far from the training home pose.")
    args = parser.parse_args()

    if args.connect_only:
        # Minimal hardware sanity: build robot config + connect + dump one obs,
        # without touching the policy server or motors. Use this BEFORE running
        # autonomous mode on a fresh setup.
        from lerobot.common.robot_devices.robots.configs import TrossenAISoloRobotConfig
        from lerobot.common.robot_devices.robots.utils import make_robot_from_config
        robot_cfg = TrossenAISoloRobotConfig(camera_interface=args.camera_interface)
        if args.follower_ip:
            robot_cfg.follower_arms["main"].ip = args.follower_ip
        if not args.with_leader:
            robot_cfg.leader_arms = {}
        elif args.leader_ip:
            robot_cfg.leader_arms["main"].ip = args.leader_ip
        if args.cam_main_serial and "cam_main" in robot_cfg.cameras:
            robot_cfg.cameras["cam_main"].serial_number = int(args.cam_main_serial)
        if args.cam_wrist_serial and "cam_wrist" in robot_cfg.cameras:
            robot_cfg.cameras["cam_wrist"].serial_number = int(args.cam_wrist_serial)
        robot = make_robot_from_config(robot_cfg)
        log.info("Connecting...")
        robot.connect()
        try:
            obs = robot.capture_observation()
            for k, v in obs.items():
                shape = getattr(v, "shape", None)
                dtype = getattr(v, "dtype", None)
                log.info("  %-40s shape=%s dtype=%s", k, shape, dtype)
        finally:
            robot.disconnect()
        return

    bridge = TrossenSoloPolicyBridge(
        server_host=args.server_host,
        server_port=args.server_port,
        control_frequency=args.control_frequency,
        rate_of_inference=args.rate_of_inference,
        max_steps=args.max_steps,
        mode=args.mode,
        camera_interface=args.camera_interface,
        send_wrist=not args.no_wrist,
        follower_ip=args.follower_ip,
        leader_ip=args.leader_ip,
        no_leader=not args.with_leader,
        cam_main_serial=args.cam_main_serial,
        cam_wrist_serial=args.cam_wrist_serial,
        reset_to_home=args.reset_to_home,
    )
    try:
        prompt = args.task_prompt if args.task_prompt is not None else _pick_task_prompt_interactively()
        bridge.run_episode(prompt)
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
