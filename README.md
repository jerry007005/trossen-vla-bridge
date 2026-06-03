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
