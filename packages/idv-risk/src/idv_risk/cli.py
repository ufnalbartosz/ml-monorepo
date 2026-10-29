"""Command line: train, evaluate, score.

    uv run python -m idv_risk.cli train --out models/candidate
    uv run python -m idv_risk.cli evaluate --model models/candidate
    uv run python -m idv_risk.cli score --model models/candidate --input apps.parquet

Every subcommand takes ``--data`` and falls back to the synthetic generator, so
the whole thing runs end to end on a fresh checkout with nothing downloaded and
no biometric data anywhere near the repository.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from idv_risk.artifact import RiskModel
from idv_risk.config import TrainingConfig
from idv_risk.pipeline import evaluate, summarise, train
from idv_risk.synthetic import GeneratorConfig, generate


def load_frame(path: Path | None, rows: int, seed: int) -> pd.DataFrame:
    """Read a table, or generate one when no path is given."""
    if path is None:
        return generate(GeneratorConfig(n_applications=rows, seed=seed))

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".csv", ".gz"):
        return pd.read_csv(path, parse_dates=["timestamp"])

    raise ValueError(f"unsupported input format: {path.suffix} (use .parquet or .csv)")


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data", type=Path, default=None, help="parquet/csv table; default synthetic"
    )
    parser.add_argument("--rows", type=int, default=12_000, help="rows to synthesise")
    parser.add_argument("--seed", type=int, default=20260801)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="idv_risk",
        description="XGBoost risk-fusion engine for identity verification decisions.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="fit a model and write a bundle")
    _add_data_arguments(train_parser)
    train_parser.add_argument("--out", type=Path, default=Path("models/idv-risk"))
    train_parser.add_argument("--max-step-up-rate", type=float, default=0.20)
    train_parser.add_argument(
        "--no-monotone",
        action="store_true",
        help="drop the schema's monotone constraints (for ablation)",
    )
    train_parser.add_argument(
        "--report", type=Path, default=None, help="write the JSON report here"
    )

    eval_parser = subparsers.add_parser(
        "evaluate", help="score a labelled table with a saved model"
    )
    _add_data_arguments(eval_parser)
    eval_parser.add_argument("--model", type=Path, required=True)
    eval_parser.add_argument("--report", type=Path, default=None)

    score_parser = subparsers.add_parser("score", help="score unlabelled applications")
    score_parser.add_argument("--model", type=Path, required=True)
    score_parser.add_argument("--input", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, default=None)

    return parser


def command_train(args: argparse.Namespace) -> int:
    frame = load_frame(args.data, args.rows, args.seed)

    config = TrainingConfig()
    if args.no_monotone:
        config = config.replace(model=config.model.__class__(use_monotone_constraints=False))

    result = train(frame, config, max_step_up_rate=args.max_step_up_rate)

    print("=== validation ===")
    print(summarise(result.validation_report))
    print("\n=== test (untouched until now) ===")
    print(summarise(result.test_report))
    print(
        f"\nthresholds: accept < {result.model.thresholds.accept_below:.4f} "
        f"<= step-up < {result.model.thresholds.reject_at_or_above:.4f} <= reject"
    )

    result.model.save(args.out)
    print(f"saved bundle to {args.out}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result.test_report, indent=2, default=str))
        print(f"wrote report to {args.report}")

    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    model = RiskModel.load(args.model)
    frame = load_frame(args.data, args.rows, args.seed)

    report = evaluate(model, frame)
    print(summarise(report))

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote report to {args.report}")

    return 0


def command_score(args: argparse.Namespace) -> int:
    model = RiskModel.load(args.model)
    frame = load_frame(args.input, rows=0, seed=0)

    scored = model.score(frame)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.suffix == ".parquet":
            scored.to_parquet(args.output)
        else:
            scored.to_csv(args.output, index=False)
        print(f"wrote {len(scored):,} scores to {args.output}")
    else:
        print(scored.to_string())

    return 0


COMMANDS = {
    "train": command_train,
    "evaluate": command_evaluate,
    "score": command_score,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
