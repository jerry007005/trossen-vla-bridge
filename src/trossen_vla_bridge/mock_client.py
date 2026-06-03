"""Send a single dummy observation to a serve_*.py server and print the
action chunk. Validates the websocket protocol end-to-end without robot HW.

Run:
    python mock_client.py --port 8000               # 1 camera, OpenVLA
    python mock_client.py --port 8000 --two-cams    # 2 cameras, OFT / pi0
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from openpi_client.websocket_client_policy import WebsocketClientPolicy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--two-cams", action="store_true",
                   help="Send both cam_main and cam_wrist (OFT / pi0 setups)")
    p.add_argument("--primary-key", default="cam_main")
    p.add_argument("--wrist-key",   default="cam_wrist")
    p.add_argument("--prompt",      default="lift the eggplant")
    p.add_argument("--n-rounds",    type=int, default=3,
                   help="How many infer() round-trips to time")
    args = p.parse_args()

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    print("metadata:", client.get_server_metadata())

    rng = np.random.default_rng(0)
    images = {
        args.primary_key: rng.integers(0, 256, (3, 224, 224), dtype=np.uint8),
    }
    if args.two_cams:
        images[args.wrist_key] = rng.integers(0, 256, (3, 224, 224), dtype=np.uint8)
    obs = {
        "state":  rng.random(7).astype(np.float32),
        "images": images,
        "prompt": args.prompt,
    }

    for r in range(args.n_rounds):
        t0 = time.perf_counter()
        resp = client.infer(obs)
        dt = (time.perf_counter() - t0) * 1000
        actions = np.asarray(resp["actions"])
        print(
            f"round {r}: client_rtt={dt:6.1f} ms  "
            f"server_timing={resp.get('server_timing')}  "
            f"actions.shape={actions.shape}  actions.dtype={actions.dtype}"
        )
        if r == 0:
            print(f"  actions[0] = {actions[0]}")


if __name__ == "__main__":
    main()
