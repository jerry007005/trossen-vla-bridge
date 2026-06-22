# trossen-vla-bridge

Bridges three fine-tuned VLAs — **OpenVLA-7B**, **OpenVLA-OFT**, and **π₀** — to a
**Trossen AI Solo** arm using the
[`openpi-client`](https://github.com/Physical-Intelligence/openpi/tree/main/packages/openpi-client)
websocket protocol.

The model server and the robot client run in **separate venvs** (each VLA has
hard dependency pins that conflict with the others). The four `scripts/setup_*.sh`
files take care of the entire dance: clone the upstream repo, build the venv,
patch the source, install the bridge.

## Architecture

```
        ┌─────────────────────────────┐         ┌────────────────────────────┐
        │   policy server (one of)    │         │  Trossen AI Solo client    │
        │  - serve_openvla.py         │ ────►   │  - client_trossen_solo.py  │
        │  - serve_oft.py             │ msg-    │      ▼                     │
        │  - serve_pi0.py             │ pack    │  lerobot ManipulatorRobot  │
        │                             │ over    │      ▼                     │
        │  Model: HF repo or local    │ websock │  trossen-arm SDK + 2 RS    │
        └─────────────────────────────┘         │  cams (cam_main, cam_wrist)│
                                                └────────────────────────────┘

      Protocol (server ⇄ client):
        REQ : {"state": (7,), "images": {"cam_main": (3,224,224)u8, ...}, "prompt": str}
        RESP: {"actions": (chunk_size, 7) f32}
```

`chunk_size` is reported in the server's metadata payload so the client can pick
its query cadence (every `chunk_size` action steps → re-query).

| Server | chunk_size | input cameras | typical RTT (H200 NVL, after warm-up) |
| --- | --- | --- | --- |
| `serve_openvla.py` | 1 | cam_main only | ~190 ms / query |
| `serve_oft.py` | 20 | cam_main + cam_wrist | ~95 ms / query |
| `serve_pi0.py` | 50 | cam_main + cam_wrist | ~48 ms / query |

The corresponding HF checkpoints (private, default repo IDs in the server scripts):
- `UoA-Trossen-Arm/openvla-7b-lift-eggplant` — merged
- `UoA-Trossen-Arm/openvla-7b-oft-lift-eggplant` — merged + L1 action head
- `UoA-Trossen-Arm/pi0-lift-eggplant-pytorch` — PyTorch port

## Layout

```
trossen-vla-bridge/
├── README.md                              <- you are here
├── pyproject.toml                         <- minimal base package (numpy, openpi-client, websockets)
├── src/trossen_vla_bridge/
│   ├── _websocket_policy_server.py        <- vendored openpi server (keepalive disabled)
│   ├── serve_openvla.py                   <- OpenVLA-7B server
│   ├── serve_oft.py                       <- OpenVLA-OFT server
│   ├── serve_pi0.py                       <- π₀ PyTorch server
│   ├── mock_client.py                     <- dummy client to verify the protocol
│   └── client_trossen_solo.py             <- real Trossen AI Solo hardware client
├── packages/openpi-client/                <- vendored, with `ping_interval=None` patch
├── patches/                               <- `git apply`-able diffs for each upstream repo
│   ├── openvla.patch                      <- forge_dataset entry in OXE configs/transforms
│   ├── openvla_oft.patch                  <- TROSSEN_CONSTANTS + forge_dataset
│   ├── openpi.patch                       <- TrossenDataConfig + pyav video backend
│   ├── openpi_trossen_policy.py           <- new file dropped into openpi/src/openpi/policies/
│   └── lerobot.patch                      <- Trossen AI Solo IPs + RealSense SNs
└── scripts/
    ├── setup_openvla_server.sh
    ├── setup_oft_server.sh
    ├── setup_pi0_server.sh
    └── setup_client.sh
```

## Install (server side)

Pick **one** VLA per machine — the deps don't co-exist.

```bash
# OpenVLA-7B (single image, simplest)
./scripts/setup_openvla_server.sh

# OpenVLA-OFT (two images, L1 head, 20-step chunks)
./scripts/setup_oft_server.sh

# π₀ (two images, 50-step chunks, fastest)
./scripts/setup_pi0_server.sh
```

Each script:
1. Clones the upstream repo into `./_envs/<name>/` (override with `ENV_ROOT=...`).
2. Builds a uv-managed venv with the right Python (3.10 / 3.10 / 3.11).
3. Upgrades torch to a Blackwell-capable wheel (`cu128`).
4. Applies the relevant patch from `./patches/` so the upstream code knows about
   `forge_dataset` and `TROSSEN_CONSTANTS`.
5. Installs `trossen-vla-bridge` so the `trossen-serve-*` CLIs are on PATH.

## Install (client side)

On the machine **physically connected to the arm**:

```bash
./scripts/setup_client.sh
```

This pulls `Interbotix/lerobot @ trossen-ai` and our default Trossen AI Solo
hardware config (follower IP `192.168.1.4`, two Intel RealSense cameras with
SNs `838212073584` / `409122274608`). If your hardware differs, edit
`patches/lerobot.patch` **before** running the script, or pass the overrides
on the CLI (see below).

## Quick start

After `setup_<X>_server.sh` and `setup_client.sh`:

### On the server box

```bash
source _envs/openpi/.venv/bin/activate          # or openvla / openvla-oft
trossen-serve-pi0   --port 8000                 # or trossen-serve-openvla / -oft
```

The first request after startup pays the torch.compile cost (10–60 s).
Subsequent requests are at the RTT shown in the table above.

### On the client box

```bash
source _envs/lerobot/.venv/bin/activate

# Step A — hardware connectivity (no server, no motion)
trossen-client --connect-only
# expect: joint state shape (7,), two RGB images at (480, 640, 3)

# Step B — dry-run with the policy server (reads obs, queries server, prints
# actions; motors stay still)
trossen-client --mode test \
    --task-prompt "lift the eggplant" \
    --server-host <server-ip> --server-port 8000 \
    --max-steps 30

# Step C — autonomous (real motion; have the E-stop ready)
trossen-client --mode autonomous \
    --task-prompt "lift the eggplant" \
    --server-host <server-ip> --server-port 8000 \
    --max-steps 60

# For pi0 add --no-anchor-offset (see "Action coord. frame" below).
```

### Sanity-check the protocol without any robot

```bash
source _envs/openpi/.venv/bin/activate    # or whichever VLA's venv
trossen-mock-client --port 8000 --two-cams --n-rounds 3
```

Sends a random observation and prints the returned action chunk; round-trips
should match the RTT table above once the model is warm.

## Hardware overrides without re-patching

```bash
trossen-client --connect-only \
    --follower-ip 192.168.1.4 \
    --cam-main-serial 838212073584 \
    --cam-wrist-serial 409122274608
```

`--with-leader` re-enables the leader arm (needed only for teleop, not for
autonomous policy execution). `--camera-interface opencv` switches off
RealSense and uses /dev/video* indices instead.

## Action coord. frame (anchor offset)

OpenVLA-7B and OpenVLA-OFT, the way we registered `forge_dataset` in their
RLDS pipeline (`ActionEncoding.EEF_POS`), emit **absolute joint targets in the
training-data coordinate frame** — i.e. they say "send joint 1 to 1.85 rad",
not "rotate joint 1 by +0.05 rad". If the physical arm you're deploying on has
a different motor-zero calibration than the arm the data was collected with
(even the same Trossen AI Solo model), those absolute targets send it to the
wrong physical pose.

π₀ avoids this because our `LeRobotTrossenDataConfig` sets
`use_delta_joint_actions=True` and uses openpi's `AbsoluteActions` output
transform: π₀ predicts deltas internally and rebuilds an absolute target by
adding the robot's current state. That makes π₀ calibration-invariant.

To get the same behaviour for OpenVLA / OFT without retraining, the client
applies an **anchor offset**: at the first inference of each episode, it
computes

```
offset = robot.actual_state - first_predicted_action
```

and adds that same constant to every subsequent action in every subsequent
chunk. The model's trajectory shape is preserved; only the absolute frame is
shifted so the first commanded target lands on the robot's true starting pose.

```bash
# default: anchor offset ON  (correct for OpenVLA / OFT)
trossen-client --mode autonomous --task-prompt "lift the eggplant" ...

# disable for π₀ (and for any other model trained directly on delta actions)
trossen-client --mode autonomous --task-prompt "lift the eggplant" \
    ... --no-anchor-offset
```

The first inference of each episode logs the offset:

```
Anchor offset (state - first predicted): [+0.45, -0.71, -0.32, -1.6, ...]
```

- Offset values of several rad on most joints ⇒ calibration mismatch was indeed
  significant; the offset is doing real work.
- Offset values < 0.1 rad on all joints ⇒ calibration was already close, and
  any remaining problem is elsewhere (joint ordering, sign convention, ...).

Limitations: the offset is fixed for the whole episode, so it cannot fix
drift that the model itself produces, nor can it correct a joint whose axis
direction is flipped. If you have a joint that the model trained to rotate
positive but on your arm rotates negative, you need a sign flip — that lives
outside this re-anchoring scheme.

## Delta-action training (`--delta-corrected`)

The anchor offset above is a deployment-time band-aid. The proper fix is to
train OpenVLA / OpenVLA-OFT on **deltas** to begin with, the same way pi0 is
trained. `patches/openvla.patch` and `patches/openvla_oft.patch` now do this
inside `forge_dataset_transform`:

```python
# Per-step pre-transform: arm joints become deltas, gripper stays absolute.
arm_delta = action[:, :6] - state[:, :6]
gripper   = action[:, 6:7]
trajectory["action"] = tf.concat([arm_delta, gripper], axis=-1)
```

The dataset_statistics recomputed during training now describe the
**delta distribution** for arm joints (≈ ±0.05 rad/step) instead of the absolute
joint range. The model learns small relative motions, identical in spirit to
pi0's `use_delta_joint_actions=True`.

At inference, the server must add the state back to reconstruct an absolute
joint target — pass `--delta-corrected` to the server:

```bash
# Server (use after retraining with the new transforms.py)
trossen-serve-openvla --checkpoint <ckpt> --port 8000 --delta-corrected
trossen-serve-oft     --checkpoint <ckpt> --port 8000 --delta-corrected

# Client (no special flag needed; the server's output is already absolute)
trossen-client --mode autonomous --task-prompt "..." --no-anchor-offset
```

Why also `--no-anchor-offset` on the client: with delta training, the model
output (after server-side state-add-back) is already a calibration-invariant
absolute target. The anchor offset would do nothing useful and could fight the
delta correction. For old (non-delta) checkpoints, leave anchor offset on and
omit `--delta-corrected`.

**Recipe to retrain**:

1. Make sure your local `_envs/openvla{,-oft}` clone has the latest patch
   applied (`./scripts/setup_*.sh` reapplies it on a clean install).
2. Delete any pre-existing `dataset_statistics.json` from the run dir if you're
   keeping the same `--run_id_note`; otherwise the loader will reuse stale
   absolute-action stats. Easiest: use a fresh `--run_id_note` like
   `--delta`.
3. Train as usual; the stats are auto-recomputed on the new (delta) data.
4. Deploy with `--delta-corrected` on the server and `--no-anchor-offset` on
   the client.

## Safety

`prismatic/.../TrossenAISoloRobotConfig` caps every motor command at
`max_relative_target=5°` from the current pose. Don't raise that until the
policy has shown it tracks your initial pose well.

Recommended progression:
1. `--connect-only` — verifies the cameras and arm SDK.
2. `--mode test` — verifies the policy server is reachable, the action shape
   matches `(chunk_size, 7)`, and the predicted joint values land within the
   training data's per-joint range (check `dataset_statistics.json` in the
   HF checkpoint repo).
3. `--mode autonomous` with `--max-steps 60` and the E-stop in reach.

## Notes

- π₀'s PyTorch path overwrites a handful of files in your transformers
  package (`models_pytorch/transformers_replace/*`). `setup_pi0_server.sh`
  does that automatically. `uv cache clean transformers` reverts it.
- The vendored `openpi-client` disables websocket keepalive ping. Without
  this, the client times out during the first torch.compile call.
- The lerobot fork used here is `Interbotix/lerobot @ trossen-ai` (v0.1.0 era).
  The newer `TrossenRobotics/lerobot_trossen` plugin is recommended by Trossen
  for new projects, but the integration with openpi-style policies isn't as
  battle-tested yet.

## Licence

MIT
