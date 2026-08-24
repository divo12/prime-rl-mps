"""Run one MPS training config end-to-end and persist per-step metrics as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prime_rl_mps.train import load_config, train


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a TOML MPSConfig file")
    parser.add_argument("--out", required=True, help="where to write the metrics JSON")
    args = parser.parse_args(argv)

    config = load_config(Path(args.config))
    metrics = train(config)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"config": config.model_dump(mode="json"), "metrics": metrics}, indent=2) + "\n")
    print("TRAINING_RUN_OK")


if __name__ == "__main__":
    main()
