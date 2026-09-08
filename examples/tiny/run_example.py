"""Run a tiny, synthetic workflow through the outcome-blind freeze boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from variantshift.outcome_lock import create_outcome_lock, freeze_predictions
from variantshift.schemas import all_single_substitutions, sequence_sha256, write_table
from variantshift.transportability import fit_transportability


def _development_table() -> pd.DataFrame:
    rows = []
    for family in range(12):
        for model_index, model in enumerate(("model-a", "model-b")):
            rows.append(
                {
                    "protocol_id": "tiny-development-v1",
                    "panel_id": "tiny-development",
                    "dataset_id": "synthetic-example",
                    "assay_id": f"assay-{family}",
                    "task_id": f"task-{family}",
                    "target_id": f"target-{family}",
                    "protein_id": f"protein-{family}",
                    "family_id": f"family-{family}",
                    "model_id": model,
                    "selection_gain_sd": 0.8 - 0.4 * model_index + 0.03 * family,
                    "protein_length": 80 + family,
                    "msa_neff": 10 + 2 * family,
                    "ensemble_disagreement": 0.1 + 0.02 * model_index,
                    "model_family": "masked" if model_index == 0 else "autoregressive",
                }
            )
    return pd.DataFrame(rows)


def run_example(output_dir: Path) -> dict[str, Path]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.DataFrame(
        {
            "panel_id": ["tiny-confirmation", "tiny-confirmation"],
            "target_id": ["tiny-a", "tiny-b"],
            "protein_id": ["tiny-a", "tiny-b"],
            "sequence": ["ACDE", "FGHI"],
            "sequence_sha256": [sequence_sha256("ACDE"), sequence_sha256("FGHI")],
            "sequence_length": [4, 4],
        }
    )
    variants = all_single_substitutions(targets)
    predictions = variants.loc[:, ["panel_id", "target_id", "variant_id"]].copy()
    predictions = pd.concat(
        [predictions.assign(model_id=model) for model in ("model-a", "model-b")],
        ignore_index=True,
    )
    predictions["protocol_id"] = "tiny-confirmation-v1"
    predictions["model_version"] = "fixture-1"
    predictions["score"] = np.arange(len(predictions), dtype=float) % 19
    predictions["status"] = "ok"
    prediction_columns = [
        "protocol_id",
        "panel_id",
        "target_id",
        "variant_id",
        "model_id",
        "model_version",
        "score",
        "status",
    ]
    paths = {
        "targets": output_dir / "targets.csv",
        "variants": output_dir / "variants.csv",
        "predictions": output_dir / "predictions.csv",
        "development": output_dir / "development.csv",
        "config": output_dir / "transport-config.json",
        "lock": output_dir / "outcome-lock.json",
    }
    write_table(targets, paths["targets"])
    write_table(variants, paths["variants"])
    write_table(predictions.loc[:, prediction_columns], paths["predictions"])
    write_table(_development_table(), paths["development"])
    paths["config"].write_text(
        json.dumps(
            {
                "numeric_features": [
                    "protein_length",
                    "msa_neff",
                    "ensemble_disagreement",
                ],
                "categorical_features": ["model_family"],
                "outer_folds": 3,
                "bootstrap_repeats": 100,
                "seed": 7,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    transport = fit_transportability(
        paths["development"], paths["config"], output_dir / "transport"
    )
    create_outcome_lock(
        paths["lock"],
        protocol_id="tiny-confirmation-v1",
        target_artifacts=[paths["targets"], paths["variants"]],
    )
    freeze_predictions(
        paths["lock"],
        prediction_artifacts=[paths["predictions"]],
        method_artifacts=[transport["bundle"], transport["method"]],
    )
    return {**paths, **{f"transport_{key}": value for key, value in transport.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    outputs = run_example(arguments.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
