"""Resumable Modal GPU runner for the outcome-sealed Domainome model panel.

Only frozen target sequences, substitution identifiers, and prevalidated structure crops are copied
into the image. Experimental outcomes are neither mounted nor downloaded by this application.
"""

# Modal's locally cached CLI currently runs on Python 3.9 and evaluates entrypoint annotations.
# Keep ``Optional`` spelling even though the remote images use Python 3.10+.
# ruff: noqa: UP045

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import modal

REPOSITORY = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/root/variantshift"
PROTOCOL_ID = "variantshift-domainome-confirmation-v1"
PANEL_VOLUME_NAME = "variantshift-panel-v1"
MODEL_CACHE_VOLUME_NAME = "variantshift-model-cache-v1"

app = modal.App("variantshift-executable-panel")
panel_volume = modal.Volume.from_name(PANEL_VOLUME_NAME, create_if_missing=True)
model_cache_volume = modal.Volume.from_name(MODEL_CACHE_VOLUME_NAME, create_if_missing=True)


def _inputs(image: modal.Image) -> modal.Image:
    return (
        image.env(
            {
                "PYTHONPATH": f"{REMOTE_ROOT}/src",
                "PYTHONHASHSEED": "0",
                "HF_HOME": "/model-cache/huggingface",
                "TORCH_HOME": "/model-cache/torch",
            }
        )
        .add_local_dir(REPOSITORY / "src", remote_path=f"{REMOTE_ROOT}/src")
        .add_local_file(
            REPOSITORY / "configs/model-panel-v1.json",
            remote_path=f"{REMOTE_ROOT}/configs/model-panel-v1.json",
        )
        .add_local_file(
            REPOSITORY / "results/confirmation/domainome-v1/targets.csv",
            remote_path=f"{REMOTE_ROOT}/results/confirmation/domainome-v1/targets.csv",
        )
        .add_local_file(
            REPOSITORY / "results/confirmation/domainome-v1/variants.csv",
            remote_path=f"{REMOTE_ROOT}/results/confirmation/domainome-v1/variants.csv",
        )
        .add_local_file(
            REPOSITORY / "results/confirmation/domainome-v1/outcome-lock.json",
            remote_path=f"{REMOTE_ROOT}/results/confirmation/domainome-v1/outcome-lock.json",
        )
        .add_local_dir(
            REPOSITORY / "artifacts/confirmation/domainome-v1/structures",
            remote_path=f"{REMOTE_ROOT}/artifacts/confirmation/domainome-v1/structures",
        )
    )


fair_esm_image = _inputs(
    modal.Image.from_registry("pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime")
    .pip_install(
        "fair-esm==2.0.0",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scipy==1.14.1",
        "scikit-learn==1.6.1",
        "biotite==0.41.2",
        "torch-geometric==2.6.1",
    )
    .run_commands(
        "python -m pip install torch-scatter==2.1.2 "
        "-f https://data.pyg.org/whl/torch-2.2.2+cu121.html"
    )
)

external_image = _inputs(
    modal.Image.from_registry("pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime")
    .apt_install("git", "wget", "tar", "ca-certificates")
    .pip_install(
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scipy==1.14.1",
        "scikit-learn==1.6.1",
        "biopython==1.81",
        "fair-esm==2.0.0",
        "transformers==4.28.1",
        "tokenizers==0.13.3",
        "datasets==2.14.7",
        "pyarrow==14.0.2",
        "huggingface-hub==0.24.7",
        "tqdm==4.66.5",
        "sentencepiece==0.2.0",
    )
    .run_commands(
        "git clone https://github.com/dauparas/ProteinMPNN.git /opt/ProteinMPNN",
        "git -C /opt/ProteinMPNN checkout 8907e6671bfbfc92303b5f79c4b5e6ce47cdef57",
        "git clone https://github.com/OATML-Markslab/Tranception.git /opt/Tranception",
        "git -C /opt/Tranception checkout 2ddf40e1db9d2d180d1b5fc9d1b39ad5b04fbb6d",
        "git clone https://github.com/westlake-repl/SaProt.git /opt/SaProt",
        "git -C /opt/SaProt checkout e91e4858b55944523f1f8d385f7b96a0d3d34c1d",
        "wget -q https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz -O /tmp/foldseek.tar.gz",
        "tar -xzf /tmp/foldseek.tar.gz -C /opt",
        "rm /tmp/foldseek.tar.gz",
    )
    .env(
        {
            "PROTEINMPNN_DIR": "/opt/ProteinMPNN",
            "TRANCEPTION_DIR": "/opt/Tranception",
            "FOLDSEEK_BINARY": "/opt/foldseek/bin/foldseek",
        }
    )
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _execute(
    model_id: str,
    target_limit: Optional[int] = None,
    target_offset: int = 0,
) -> dict[str, object]:
    import os
    import platform
    import shutil
    import time

    import pandas as pd
    import torch

    os.chdir(REMOTE_ROOT)
    from variantshift.model_adapters import (
        adapter_from_specification,
        load_model_specifications,
        score_panel,
    )
    from variantshift.schemas import write_table

    root = Path(REMOTE_ROOT)
    config = root / "configs/model-panel-v1.json"
    targets_path = root / "results/confirmation/domainome-v1/targets.csv"
    variants_path = root / "results/confirmation/domainome-v1/variants.csv"
    lock_path = root / "results/confirmation/domainome-v1/outcome-lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("state") != "targets_frozen":
        raise RuntimeError(
            f"Outcome-sealed scoring requires targets_frozen, found {lock.get('state')}"
        )
    specifications = load_model_specifications(config)
    matching = [item for item in specifications if item.model_id == model_id]
    if len(matching) != 1:
        raise ValueError(f"Expected one model specification for {model_id}")
    specification = matching[0]
    targets = pd.read_csv(targets_path).sort_values("target_id").reset_index(drop=True)
    if target_offset < 0 or target_offset >= len(targets):
        raise ValueError(f"target_offset must be between 0 and {len(targets) - 1}")
    targets = targets.iloc[target_offset:].copy()
    if target_limit is not None:
        if target_limit < 1:
            raise ValueError("target_limit must be positive")
        targets = targets.head(target_limit).copy()
    target_ids = set(targets["target_id"].astype(str))
    variants = pd.read_csv(variants_path)
    variants = variants.loc[variants["target_id"].astype(str).isin(target_ids)].copy()

    adapter = adapter_from_specification(specification)
    started = time.time()
    predictions, audit = score_panel(
        adapter,
        targets,
        variants,
        protocol_id=PROTOCOL_ID,
        cache_dir=Path("/vol/cache/domainome-v1"),
    )
    elapsed = time.time() - started
    temporary = Path("/tmp") / f"variantshift-{model_id}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    prediction_path = write_table(predictions, temporary / "predictions.csv.gz")
    audit_path = write_table(audit, temporary / "prediction-audit.csv.gz")
    provenance_path = temporary / "model-provenance.json"
    provenance_path.write_text(
        json.dumps({model_id: adapter.provenance()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    passing = sorted(
        audit.loc[audit["status"].eq("ok") & audit["coverage"].ge(0.95), "target_id"]
        .astype(str)
        .tolist()
    )
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": model_id,
        "family": specification.family,
        "execution_status": "complete",
        "target_count_requested": int(targets["target_id"].nunique()),
        "target_offset": target_offset,
        "targets_at_95pct_coverage": len(passing),
        "shared_target_candidate_sha256": hashlib.sha256(
            "\n".join(passing).encode("utf-8")
        ).hexdigest(),
        "prediction_rows": len(predictions),
        "variant_rows_requested": len(variants),
        "elapsed_seconds": elapsed,
        "hardware": {
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "platform": platform.platform(),
        },
        "inputs": {
            "model_config_sha256": _sha256(config),
            "targets_sha256": _sha256(targets_path),
            "variants_sha256": _sha256(variants_path),
            "outcome_lock_sha256": _sha256(lock_path),
        },
        "artifacts": {
            prediction_path.name: _sha256(prediction_path),
            audit_path.name: _sha256(audit_path),
            provenance_path.name: _sha256(provenance_path),
        },
        "outcomes_accessed": False,
        "qualification_status": "not_started",
    }
    manifest_path = temporary / "execution-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if target_offset or target_limit is not None:
        last_offset = target_offset + len(targets) - 1
        destination = (
            Path("/vol/executions/domainome-v1")
            / model_id
            / "shards"
            / f"{target_offset:04d}-{last_offset:04d}"
        )
    else:
        destination = Path("/vol/executions/domainome-v1") / model_id
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(temporary, destination)
    panel_volume.commit()
    return manifest


@app.function(
    image=fair_esm_image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={"/vol": panel_volume, "/model-cache": model_cache_volume},
)
def execute_fair_esm(
    model_id: str,
    target_limit: Optional[int] = None,
    target_offset: int = 0,
) -> dict[str, object]:
    if model_id not in {"esm2_650m", "esm_if1"}:
        raise ValueError(f"Unsupported fair-esm Modal model: {model_id}")
    return _execute(model_id, target_limit, target_offset)


@app.function(
    image=external_image,
    gpu="L4",
    cpu=8,
    memory=49152,
    timeout=24 * 60 * 60,
    volumes={"/vol": panel_volume, "/model-cache": model_cache_volume},
)
def execute_external(
    model_id: str,
    target_limit: Optional[int] = None,
    target_offset: int = 0,
) -> dict[str, object]:
    if model_id not in {"proteinmpnn", "saprot_35m", "tranception_l_no_retrieval"}:
        raise ValueError(f"Unsupported external Modal model: {model_id}")
    return _execute(model_id, target_limit, target_offset)


@app.function(
    image=external_image,
    cpu=4,
    memory=16384,
    timeout=60 * 60,
    volumes={"/vol": panel_volume},
)
def merge_shards(model_id: str, expected_targets: int = 426) -> dict[str, object]:
    """Merge disjoint target shards into one execution artifact without qualification."""
    import os
    import shutil

    import pandas as pd

    os.chdir(REMOTE_ROOT)
    from variantshift.model_adapters import load_model_specifications
    from variantshift.schemas import PREDICTION_SCHEMA, write_table

    root = Path(REMOTE_ROOT)
    source = Path("/vol/executions/domainome-v1") / model_id / "shards"
    shard_directories = sorted(path for path in source.iterdir() if path.is_dir())
    if not shard_directories:
        raise RuntimeError(f"No execution shards found for {model_id}")
    predictions = pd.concat(
        [pd.read_csv(path / "predictions.csv.gz") for path in shard_directories],
        ignore_index=True,
    ).drop_duplicates(["target_id", "variant_id"], keep="last")
    audits = pd.concat(
        [pd.read_csv(path / "prediction-audit.csv.gz") for path in shard_directories],
        ignore_index=True,
    ).drop_duplicates("target_id", keep="last")
    if audits["target_id"].nunique() != expected_targets:
        raise RuntimeError(
            f"Expected {expected_targets} audited targets for {model_id}, found "
            f"{audits['target_id'].nunique()}"
        )
    PREDICTION_SCHEMA.validate(predictions)
    specification = next(
        item
        for item in load_model_specifications(root / "configs/model-panel-v1.json")
        if item.model_id == model_id
    )
    passing = sorted(
        audits.loc[
            audits["status"].eq("ok") & audits["coverage"].ge(0.95), "target_id"
        ]
        .astype(str)
        .tolist()
    )
    temporary = Path("/tmp") / f"variantshift-{model_id}-merged"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    prediction_path = write_table(predictions, temporary / "predictions.csv.gz")
    audit_path = write_table(audits, temporary / "prediction-audit.csv.gz")
    provenance_path = temporary / "model-provenance.json"
    shutil.copy2(shard_directories[0] / "model-provenance.json", provenance_path)
    manifests = [
        json.loads((path / "execution-manifest.json").read_text(encoding="utf-8"))
        for path in shard_directories
    ]
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "model_id": model_id,
        "family": specification.family,
        "execution_status": "complete",
        "target_count_requested": expected_targets,
        "targets_at_95pct_coverage": len(passing),
        "shared_target_candidate_sha256": hashlib.sha256(
            "\n".join(passing).encode("utf-8")
        ).hexdigest(),
        "prediction_rows": len(predictions),
        "elapsed_gpu_seconds_sum": sum(
            float(item["elapsed_seconds"]) for item in manifests
        ),
        "execution_shards": [path.name for path in shard_directories],
        "hardware": [item["hardware"] for item in manifests],
        "inputs": manifests[0]["inputs"],
        "artifacts": {
            prediction_path.name: _sha256(prediction_path),
            audit_path.name: _sha256(audit_path),
            provenance_path.name: _sha256(provenance_path),
        },
        "outcomes_accessed": False,
        "qualification_status": "not_started",
    }
    manifest_path = temporary / "execution-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    destination = Path("/vol/executions/domainome-v1") / model_id
    for artifact in temporary.iterdir():
        shutil.copy2(artifact, destination / artifact.name)
    panel_volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    model_id: str,
    target_limit: Optional[int] = None,
    target_offset: int = 0,
    merge: bool = False,
) -> None:
    if merge:
        result = merge_shards.remote(model_id)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    function = execute_fair_esm if model_id in {"esm2_650m", "esm_if1"} else execute_external
    result = function.remote(model_id, target_limit, target_offset)
    print(json.dumps(result, indent=2, sort_keys=True))
