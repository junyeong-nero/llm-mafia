from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

from src.metrics.report import to_report_text
from src.metrics.comparison import comparison_report
from src.config import AppConfig, load_config
from src.runner.match_runner import RunnerType, run_match
from src.runner.tournament import run_tournament


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
        "--web",
        action="store_true",
        help="Run the web dashboard instead of a CLI match",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host binding for the web dashboard (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the web dashboard (default: 8000)",
    )
    parser.add_argument(
        "--runner",
        choices=("legacy", "graph"),
        default="graph",
        help="Runner implementation to use (default: graph)",
    )
    parser.add_argument(
        "--tournament",
        type=int,
        metavar="MATCHES",
        help="Compare models with balanced seat rotation (equal counts, full model cycles required)",
    )
    return parser


def _print_config_summary(config: AppConfig) -> None:
    model_summary = ", ".join(
        f"{model.name}:{model.provider or config.llm.provider}/{model.model} x{model.count}"
        for model in config.llm.models
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


def _run_web_dashboard(host: str, port: int) -> None:
    import uvicorn

    from src.web.server import create_app

    print(f"Starting web dashboard at http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port)


def _print_tournament_progress(payload: dict[str, object]) -> None:
    if payload["kind"] == "tournament_match":
        print(
            f"Match {payload['match']}/{payload['total']} "
            f"seed={payload['seed']} rotation={payload['rotation']} "
            f"summary={payload['summary_path']}",
            flush=True,
        )
    elif payload["kind"] in {"phase", "game_end", "provider_retry"}:
        print(payload, flush=True)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.web and args.tournament is not None:
        parser.error("--tournament cannot be combined with --web")
    if args.web:
        _run_web_dashboard(args.host, args.port)
        return

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    _print_config_summary(config)
    if args.tournament is not None:
        try:
            tournament = run_tournament(
                config,
                rounds=args.tournament,
                seed=args.seed if args.seed is not None else 42,
                max_rounds=args.max_rounds,
                runner=cast(RunnerType, args.runner),
                balanced=True,
                progress_callback=_print_tournament_progress,
            )
        except (ValueError, RuntimeError) as exc:
            raise SystemExit(str(exc)) from exc
        print(comparison_report(tournament.model_stats))
        print(f"- summary: {tournament.summary_path}")
        return
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
