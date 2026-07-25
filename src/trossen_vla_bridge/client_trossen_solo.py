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

    # EEF-delta mode for the new Bridge-V2-style ckpt
    # (openvla-7b-trossen-pnp-4diverse-eef-png). state is [xyz, rpy, gripper];
    # action[:6] is EE delta (clipped to +/- 0.01), action[6] > 0.5 -> open.
    python client_trossen_solo.py --mode autonomous \
        --action-mode eef_delta --task-prompt "..."
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import time
from collections import defaultdict
from datetime import datetime
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
        action_mode: str = "joint_abs",
        eef_goal_time: float = 0.1,
        record_dir: str | None = None,
        max_relative_target: float | list[float] | None = 0.3,
        black_wrist: bool = False,
        black_primary: bool = False,
    ) -> None:
        if mode not in {"autonomous", "test"}:
            raise ValueError(f"mode must be 'autonomous' or 'test'; got {mode!r}")
        if action_mode not in {"joint_abs", "eef_delta"}:
            raise ValueError(f"action_mode must be 'joint_abs' or 'eef_delta'; got {action_mode!r}")
        self.mode = mode
        self.control_frequency = control_frequency
        self.dt = 1.0 / control_frequency
        self.rate_of_inference = rate_of_inference
        self.max_steps = max_steps
        self.send_wrist = send_wrist
        # When True, still advertise cam_wrist to the server but feed an all-zeros
        # (pure black) frame instead of the real wrist view -- lets you mask the
        # wrist input for a 2-camera checkpoint without dropping the key entirely.
        self.black_wrist = black_wrist
        # Same idea for the primary view: send cam_main as an all-zeros frame
        # (the real frame is still read to size the black image).
        self.black_primary = black_primary
        self.reset_to_home = reset_to_home
        self.action_mode = action_mode
        self.eef_goal_time = eef_goal_time
        # Track last gripper open/close so we don't re-issue the same
        # blocking(=False) gripper command every 50ms in eef_delta mode.
        self._last_gripper_open: bool | None = None

        # Per-step safety clamp on joint motion, in the arm's NATIVE units
        # (radians for the Trossen driver). Each control step the commanded goal
        # is capped so no joint moves more than this far from its current
        # measured position -- our own copy of lerobot's ensure_safe_goal_position
        # so it survives the eventual lerobot_trossen migration and is controlled
        # from here in the right (radian) units. None disables it. Only applied
        # in joint_abs mode; eef_delta bounds its own +/- 0.01 cartesian step.
        self.max_relative_target = max_relative_target
        self._clamp_warn_count = 0

        # Optional per-camera video recording. Writers are created lazily on the
        # first frame (we need the frame's HxW to size the VideoWriter) and keyed
        # by camera name. Left empty/None when --record-dir is not passed.
        self.record_dir = record_dir
        self._video_writers: dict[str, "cv2.VideoWriter"] = {}
        self._video_paths: dict[str, str] = {}
        if self.record_dir is not None:
            os.makedirs(self.record_dir, exist_ok=True)
            self._record_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log.info("Recording camera streams to %s (fps=%d)",
                     self.record_dir, control_frequency)

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

        # We enforce the relative-target clamp ourselves in _send_action (in the
        # arm's native radian units), so disable lerobot's built-in scalar clamp
        # to keep a single authority and avoid a redundant Present_Position read
        # per step. Its default (5.0) is inherited from the degree-based feetech
        # arms and is a no-op in radians anyway. When our clamp is disabled we
        # leave lerobot's config default untouched as a backstop.
        if self.max_relative_target is not None:
            robot_cfg.max_relative_target = None

        self.robot = make_robot_from_config(robot_cfg)
        self.robot.connect()

        self.episode_step = 0
        self.current_chunk: np.ndarray | None = None
        self.chunk_idx = 0
        # buffer of (step -> list of candidate actions), used for temporal
        # ensembling if you later switch it on.
        self.action_buffer: Dict[int, List[np.ndarray]] = defaultdict(list)

    # ------------------------------------------------------------------ obs

    def _read_obs(self, observation_dict: dict | None = None) -> dict:
        # The caller (the run loop) already captures once per control step so it
        # can record every frame; on inference steps it hands us that same
        # observation to avoid a second camera read. _move_to_home() calls this
        # with no argument, so capture on demand when none is supplied.
        if observation_dict is None:
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
        joint_state = np.asarray(joint_vals[:ACTION_DIM], dtype=np.float32)

        # Choose which state to advertise to the server based on action_mode:
        #  joint_abs -> 7 raw joint positions (backward compatible)
        #  eef_delta -> [xyz(3), rpy(3), gripper_joint(1)] built via raw driver
        #               (matches training-time state_obs_keys=[EEF_state, gripper_state]).
        if self.action_mode == "eef_delta":
            from scipy.spatial.transform import Rotation as _R
            raw = self._raw_driver()
            pos = np.asarray(raw.get_cartesian_positions(), dtype=np.float32)  # [xyz, rotvec]
            xyz = pos[:3]
            rpy = _R.from_rotvec(pos[3:]).as_euler("xyz")
            state = np.concatenate([xyz, rpy, joint_state[6:7]]).astype(np.float32)
        else:
            state = joint_state

        return {
            "state":  state,
            "images": self._extract_images(observation_dict),
            "prompt": getattr(self, "_task_prompt", "lift the eggplant"),
        }

    def _extract_images(self, observation_dict: dict) -> dict[str, np.ndarray]:
        # lerobot returns HWC RGB. Send RAW 480x640 CHW to the server -- the OFT
        # and openvla servers will cv2-resize to 224x224 internally to match the
        # tf.image.resize pixel-level behaviour their training data went through.
        # The pi0 server keeps the raw frame as-is and lets openpi's image
        # pipeline handle it.
        images: dict[str, np.ndarray] = {}
        # In black-wrist mode we skip reading the real cam_wrist frame and
        # substitute an all-zeros image below (sized to cam_main).
        wanted = ["cam_main"]
        if self.send_wrist and not self.black_wrist:
            wanted.append("cam_wrist")
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
        # Pure-black primary frame (real frame was read above only to size it).
        if self.black_primary:
            images["cam_main"] = np.zeros_like(images["cam_main"])
        # Pure-black wrist frame, same CHW shape/dtype as cam_main. Sent under
        # the cam_wrist key so a 2-camera server still gets both inputs.
        if self.black_wrist:
            images["cam_wrist"] = np.zeros_like(images["cam_main"])
        return images

    # -------------------------------------------------------------- recording

    def _record_frames(self, images: dict[str, np.ndarray]) -> None:
        """Append the current frame of every camera to its own mp4.

        `images[cam]` is RGB, CHW, uint8 (exactly what gets sent to the server).
        VideoWriter wants HWC BGR, so we transpose + colour-convert here. Writers
        are created on first use, sized to the frame that camera actually
        produced; they are released in shutdown().
        """
        for cam, chw in images.items():
            hwc_rgb = np.transpose(chw, (1, 2, 0))
            bgr = cv2.cvtColor(hwc_rgb, cv2.COLOR_RGB2BGR)
            writer = self._video_writers.get(cam)
            if writer is None:
                h, w = bgr.shape[:2]
                path = os.path.join(
                    self.record_dir, f"{self._record_stamp}_{cam}.mp4"
                )
                # The run loop records one frame per control step, so the frame
                # rate equals control_frequency and the clip plays back at true
                # real-time speed.
                writer = cv2.VideoWriter(
                    path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(self.control_frequency),
                    (w, h),
                )
                if not writer.isOpened():
                    log.warning("Could not open VideoWriter for %s at %s; "
                                "disabling recording for this camera", cam, path)
                    self._video_writers[cam] = writer  # cache the dud so we stop retrying
                    continue
                self._video_writers[cam] = writer
                self._video_paths[cam] = path
                log.info("  recording %s -> %s (%dx%d)", cam, path, w, h)
            if writer.isOpened():
                writer.write(bgr)

    # ------------------------------------------------------------------ act

    def _raw_driver(self):
        """Return the underlying trossen_arm.TrossenArmDriver used by lerobot.

        Only reachable after `self.robot.connect()`; needed for EEF-delta mode
        because lerobot's public `send_action(tensor)` API is joint-space only.
        """
        try:
            return self.robot.follower_arms["main"].driver
        except (AttributeError, KeyError) as exc:
            raise RuntimeError(
                "Could not reach raw trossen_arm driver via lerobot -- "
                "EEF-delta mode requires the standard TrossenAISoloRobotConfig "
                "with a follower arm named 'main'."
            ) from exc

    def _read_present_position(self) -> np.ndarray:
        """Current measured follower joint positions (radians), concatenated in
        motor order across all follower arms and truncated to ACTION_DIM."""
        vals: list[float] = []
        for name in self.robot.follower_arms:
            pos = self.robot.follower_arms[name].read("Present_Position")
            vals.extend(np.asarray(pos, dtype=np.float32).reshape(-1).tolist())
        return np.asarray(vals[:ACTION_DIM], dtype=np.float32)

    def _clamp_action(self, action: np.ndarray, present: np.ndarray) -> np.ndarray:
        """Cap the per-joint magnitude of (action - present) so no joint is asked
        to move more than `max_relative_target` in one control step. Mirrors
        lerobot's ensure_safe_goal_position; `max_relative_target` is a scalar or
        an ACTION_DIM list, in radians (the Trossen driver's native units)."""
        if self.max_relative_target is None:
            return action
        limit = np.asarray(self.max_relative_target, dtype=np.float32)
        diff = action - present
        safe_diff = np.clip(diff, -limit, limit)
        safe = (present + safe_diff).astype(np.float32)
        if not np.allclose(diff, safe_diff, atol=1e-6):
            # Throttle: clamping every step at 20 Hz would flood the log.
            self._clamp_warn_count += 1
            if self._clamp_warn_count == 1 or self._clamp_warn_count % 20 == 0:
                log.warning(
                    "[clamp] step commanded a large joint move (count=%d); "
                    "requested Δ=%s -> clamped Δ=%s (limit=%s rad)",
                    self._clamp_warn_count,
                    np.round(diff, 4).tolist(),
                    np.round(safe_diff, 4).tolist(),
                    limit.tolist() if limit.ndim else float(limit),
                )
        return safe

    def _send_action(self, action: np.ndarray) -> None:
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"action shape must be ({ACTION_DIM},); got {action.shape}")
        if self.mode == "test":
            log.debug("test-mode action (not sent): %s", np.round(action, 4))
            return

        if self.action_mode == "eef_delta":
            self._send_action_eef_delta(action)
            return

        # joint_abs mode (default, backward compatible)
        # Safety clamp: cap each joint's step relative to its CURRENT measured
        # position before the goal ever reaches the motors. (eef_delta returns
        # above and bounds its own +/- 0.01 cartesian step.)
        if self.max_relative_target is not None:
            action = self._clamp_action(action, self._read_present_position())
        # The follower arm expects an action under the key
        # `action.{joint_name}` per joint; lerobot 0.1.0's ManipulatorRobot
        # consumes a torch.tensor of shape (ACTION_DIM,).
        import torch
        self.robot.send_action(torch.from_numpy(action.astype(np.float32)))

    def _send_action_eef_delta(self, action: np.ndarray) -> None:
        """EEF-delta action: action[:6] are per-step delta (dx dy dz drx dry drz)
        added to the current EE pose; action[6] > 0.5 -> open, else close.

        Mirrors sample.py::WidowXAIInterface.step_action:
          - clip arm delta to +/- 0.01 to bound each step
          - convert rpy target back to rotvec for set_cartesian_positions
          - use non-blocking calls (own loop paces at self.dt)
        """
        from scipy.spatial.transform import Rotation as _R
        from trossen_arm import ArrayDouble6, InterpolationSpace

        raw = self._raw_driver()
        cur = np.asarray(raw.get_cartesian_positions(), dtype=np.float32)  # [xyz, rotvec]
        cur_rpy = _R.from_rotvec(cur[3:]).as_euler("xyz")

        arm_delta = np.clip(action[:6].astype(np.float32), -0.01, 0.01)
        goal_xyz = cur[:3] + arm_delta[:3]
        goal_rpy = cur_rpy + arm_delta[3:]
        goal_rotvec = _R.from_euler("xyz", goal_rpy).as_rotvec()
        goal6 = np.concatenate([goal_xyz, goal_rotvec]).astype(np.float64)

        raw.set_cartesian_positions(
            ArrayDouble6(goal6),
            interpolation_space=InterpolationSpace.joint,
            goal_time=self.eef_goal_time,
            blocking=False,
        )

        want_open = bool(action[6] > 0.5)
        if self._last_gripper_open is not want_open:
            # sample.py convention: +0.04 open, -0.04 close.
            raw.set_gripper_position(
                0.04 if want_open else -0.04,
                goal_time=1.0,
                blocking=False,
            )
            self._last_gripper_open = want_open

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

            # Capture once per control step. When recording, this gives us a
            # frame for every step (smooth real-time video at control_frequency)
            # rather than only on inference steps. The captured observation is
            # reused for inference below, so inference steps read the camera once.
            step_obs_dict = None
            if self.record_dir is not None:
                step_obs_dict = self.robot.capture_observation()
                self._record_frames(self._extract_images(step_obs_dict))

            need_new_chunk = (
                self.current_chunk is None
                or self.chunk_idx >= self.rate_of_inference
            )
            if need_new_chunk:
                obs = self._read_obs(step_obs_dict)
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
        for cam, writer in self._video_writers.items():
            try:
                writer.release()
                log.info("  finalised recording for %s", cam)
            except Exception as exc:
                log.warning("Releasing VideoWriter for %s raised: %s", cam, exc)
        # The OpenCV bundled with this venv can only encode mpeg4 part 2 (mp4v),
        # which browsers / VS Code / most default players refuse to play. The
        # system ffmpeg does have libx264, so re-encode each finished file to
        # H.264 in place. If ffmpeg is missing we just leave the mp4v files.
        self._transcode_recordings_to_h264()
        try:
            self.robot.disconnect()
        except Exception as exc:
            log.warning("Robot disconnect raised: %s", exc)

    def _transcode_recordings_to_h264(self) -> None:
        if not self._video_paths:
            return
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            log.warning("ffmpeg not on PATH; leaving recordings as mp4v "
                        "(open them in VLC/mpv, or transcode manually)")
            return
        for cam, path in self._video_paths.items():
            if not os.path.exists(path):
                continue
            tmp = path + ".h264.tmp.mp4"
            cmd = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-i", path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", tmp,
            ]
            try:
                subprocess.run(cmd, check=True)
                os.replace(tmp, path)  # keep the clean <stamp>_<cam>.mp4 name
                log.info("  transcoded %s to H.264", path)
            except Exception as exc:
                log.warning("H.264 transcode of %s failed (%s); keeping mp4v file",
                            path, exc)
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass


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


def _parse_max_relative_target(s: str | None) -> float | list[float] | None:
    """Parse the --max-relative-target CLI value: 'none'/'off'/'' or a
    non-positive scalar disables it; a comma-separated string gives per-joint
    limits (must be ACTION_DIM values); otherwise a single float."""
    if s is None:
        return None
    t = s.strip().lower()
    if t in {"none", "off", "disable", ""}:
        return None
    if "," in t:
        parts = [float(x) for x in t.split(",")]
        if len(parts) != ACTION_DIM:
            raise argparse.ArgumentTypeError(
                f"--max-relative-target list must have {ACTION_DIM} values, got {len(parts)}"
            )
        return parts
    val = float(t)
    return val if val > 0 else None


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
                        help="Send only cam_main, dropping the cam_wrist key entirely "
                             "(use when serving plain OpenVLA).")
    parser.add_argument("--black-wrist", action="store_true",
                        help="Still send the cam_wrist key but with an all-zeros (pure "
                             "black) frame instead of the real wrist view. Use to mask "
                             "the wrist input for a 2-camera checkpoint (OFT/pi0) without "
                             "dropping the key. Mutually exclusive with --no-wrist.")
    parser.add_argument("--black-primary", action="store_true",
                        help="Send the cam_main key with an all-zeros (pure black) frame "
                             "instead of the real primary view. The real frame is still "
                             "read to size the black image. Can be combined with "
                             "--black-wrist to blank both views.")
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
    parser.add_argument("--record-dir", default=None,
                        help="If set, record each camera stream to its own mp4 in "
                             "this directory (files named <timestamp>_<cam>.mp4, e.g. "
                             "20260715_143022_cam_main.mp4). Frames are the RAW frames "
                             "sent to the server; fps = --control-frequency. Off by "
                             "default.")
    parser.add_argument("--max-relative-target", default="0.1",
                        help="Per-step safety clamp on joint motion, in the arm's "
                             "NATIVE units (RADIANS for the Trossen driver). Each "
                             "control step the commanded goal is capped so no joint "
                             "moves more than this far from its current MEASURED "
                             "position (our copy of lerobot's ensure_safe_goal_position). "
                             "Pass a single float for all 7 joints, a comma-separated "
                             "list of 7 for per-joint limits, or 'none' to disable. "
                             "Default 0.3 rad (~17 deg/step; ~5.7 rad/s at 20 Hz) -- "
                             "loose enough not to touch normal motion, tight enough to "
                             "stop a pathological jump. Only active in --mode autonomous "
                             "and --action-mode joint_abs (eef_delta bounds its own step).")
    parser.add_argument("--reset-to-home", action="store_true",
                        help="Before the policy loop starts, linearly interpolate from "
                             "the current pose to the trained home pose "
                             "(j1=pi/3, j2=pi/6, j3=0.6284, rest=0, gripper open) over "
                             "3 seconds. Off by default -- enable when starting a fresh "
                             "episode and the arm is far from the training home pose. "
                             "Ignored in --action-mode eef_delta (joint-space reset "
                             "doesn't apply once the policy speaks EE deltas).")
    parser.add_argument("--action-mode", choices=["joint_abs", "eef_delta"],
                        default="joint_abs",
                        help="Semantic of what obs['state'] is and what action[t] means. "
                             "'joint_abs' (default, backward compatible): state = 7 raw "
                             "joint positions, action = 7 absolute joint targets sent via "
                             "lerobot's send_action. 'eef_delta' (Bridge-V2 / new EEF "
                             "openvla ckpt): state = [xyz(3), rpy(3), joint_gripper(1)] "
                             "built via raw trossen_arm.get_cartesian_positions(); "
                             "action = [dx, dy, dz, drx, dry, drz, gripper] with arm "
                             "dims added to the current EE pose (clipped to +/- 0.01) "
                             "and dispatched via set_cartesian_positions; gripper > 0.5 "
                             "opens (+0.04), else closes (-0.04).")
    parser.add_argument("--eef-goal-time", type=float, default=0.1,
                        help="goal_time (seconds) passed to set_cartesian_positions in "
                             "eef_delta mode. Longer smooths the trajectory but tolerates "
                             "less new-command overshoot. Only used when --action-mode "
                             "eef_delta.")
    args = parser.parse_args()

    if args.no_wrist and args.black_wrist:
        parser.error("--no-wrist and --black-wrist are mutually exclusive: "
                     "--no-wrist drops cam_wrist, --black-wrist sends it black.")

    if args.action_mode == "eef_delta" and args.reset_to_home:
        log.warning("--reset-to-home is a joint-space maneuver; ignoring in eef_delta mode")
        args.reset_to_home = False

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
        action_mode=args.action_mode,
        eef_goal_time=args.eef_goal_time,
        record_dir=args.record_dir,
        max_relative_target=_parse_max_relative_target(args.max_relative_target),
        black_wrist=args.black_wrist,
        black_primary=args.black_primary,
    )
    try:
        prompt = args.task_prompt if args.task_prompt is not None else _pick_task_prompt_interactively()
        bridge.run_episode(prompt)
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
