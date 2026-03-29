from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import cast

from src.metrics.report import to_report_text
from src.config import AppConfig, load_config
from src.runner.match_runner import RunnerType, run_match


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run llm-mafia with YAML config.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic match run (default: random per run)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=10,
        help="Maximum match rounds",
    )
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Run the Streamlit dashboard instead of a CLI match",
    )
    parser.add_argument(
        "--streamlit-app",
        type=Path,
        default=Path("src/streamlit_app.py"),
        help="Path to Streamlit app file (default: src/streamlit_app.py)",
    )
    parser.add_argument(
        "--runner",
        choices=("legacy", "graph"),
        default="graph",
        help="Runner implementation to use (default: graph)",
    )
    return parser


def _print_config_summary(config: AppConfig) -> None:
    model_summary = ", ".join(
        f"{model.name}:{model.model} x{model.count}" for model in config.llm.models
    )
    print("Loaded game configuration")
    print(f"- provider: {config.llm.provider}")
    print(f"- players: {config.game.player_count}")
    print(
        "- roles: "
        f"mafia={config.game.roles.mafia}, "
        f"police={config.game.roles.police}, "
        f"doctor={config.game.roles.doctor}, "
        f"citizen={config.game.roles.citizen}"
    )
    print(f"- models: {model_summary}")


def _run_streamlit_dashboard(app_path: Path) -> None:
    if not app_path.is_file():
        raise SystemExit(f"Streamlit app not found: {app_path}")

    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def main() -> None:
    args = _build_parser().parse_args()
    if args.streamlit:
        _run_streamlit_dashboard(args.streamlit_app)
        return

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    _print_config_summary(config)
    result = run_match(
        config,
        seed=args.seed,
        max_rounds=args.max_rounds,
        runner=cast(RunnerType, args.runner),
    )
    print(to_report_text(result.metrics))
    print(f"- logs: {result.output_dir}")
    print(f"- events: {result.events_path}")
    print(f"- summary: {result.summary_path}")


if __name__ == "__main__":
    main()
