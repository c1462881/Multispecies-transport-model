#!/usr/bin/env python3
"""
main.py

Short YAML-driven entry point.
"""

import argparse

from configuration import load_yaml_config
from initialization import build_initial_state
from output import setup_output_dirs, RunLogger, Timer
from time_loop import run_time_loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        required=True,
        help="YAML file containing all run parameters.",
    )
    args = ap.parse_args()

    cfg = load_yaml_config(args.config)
    paths = setup_output_dirs(cfg)

    L = RunLogger(paths["out_dir"])
    timer = Timer(cfg.run.profile)

    try:
        state = build_initial_state(cfg, paths, L)
        run_time_loop(state, cfg, paths, timer, L)
        timer.report(L)
    finally:
        L.save()


if __name__ == "__main__":
    main()