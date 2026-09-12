"""Fail-closed Modal runner for independent model qualification executions."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import modal

# The locally cached Modal CLI evaluates annotations with Python 3.9.
# ruff: noqa: UP045

REMOTE_ROOT = "/root/variantshift"
_LOCAL_REPOSITORY = Path(__file__).resolve().parents[1]
REPOSITORY = (
    _LOCAL_REPOSITORY
    if (_LOCAL_REPOSITORY / "containers/qualification-lock-v1.json").is_file()
    else Path(REMOTE_ROOT)
)
QUALIFICATION_VOLUME = "variantshift-qualification-v1"
QUALIFICATION_CACHE_VOLUME = "variantshift-qualification-cache-v1"
MODEL_CACHE_VOLUME = "variantshift-model-cache-v1"

qualification_volume = modal.Volume.from_name(QUALIFICATION_VOLUME, create_if_missing=True)
qualification_cache = modal.Volume.from_name(
    QUALIFICATION_CACHE_VOLUME, create_if_missing=True
)
model_cache = modal.Volume.from_name(MODEL_CACHE_VOLUME, create_if_missing=True)
app = modal.App("variantshift-model-qualification")

LOCK_PATH = REPOSITORY / "containers/qualification-lock-v1.json"
CONTAINER_LOCK = json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _container_sha256(name: str) -> str:
    return _canonical_sha256(
        {
            "base_image": CONTAINER_LOCK["base_image"],
            "image": name,
            **CONTAINER_LOCK["images"][name],
        }
    )


def _inputs(image: modal.Image, container_name: str) -> modal.Image:
    return (
        image.env(
            {
                "PYTHONPATH": f"{REMOTE_ROOT}/src",
                "PYTHONHASHSEED": "0",
                "HF_HOME": "/model-cache/huggingface",
                "TORCH_HOME": "/model-cache/torch",
                "VARIANTSHIFT_CONTAINER_SHA256": _container_sha256(container_name),
                "VARIANTSHIFT_CONTAINER_KIND": container_name,
            }
        )
        .add_local_dir(REPOSITORY / "src", remote_path=f"{REMOTE_ROOT}/src")
        .add_local_file(
            REPOSITORY / "workflow/modal_qualification.py",
            remote_path=f"{REMOTE_ROOT}/workflow/modal_qualification.py",
        )
        .add_local_file(
            REPOSITORY / "configs/model-panel-v1.json",
            remote_path=f"{REMOTE_ROOT}/configs/model-panel-v1.json",
        )
        .add_local_file(
            REPOSITORY / "configs/qualification-v1.json",
            remote_path=f"{REMOTE_ROOT}/configs/qualification-v1.json",
        )
        .add_local_file(
            LOCK_PATH,
            remote_path=f"{REMOTE_ROOT}/containers/qualification-lock-v1.json",
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
        .add_local_file(
            REPOSITORY / "artifacts/qualification/proteingym-parity-v1/targets.csv",
            remote_path=f"{REMOTE_ROOT}/artifacts/qualification/proteingym-parity-v1/targets.csv",
        )
        .add_local_file(
            REPOSITORY / "artifacts/qualification/proteingym-parity-v1/variants.csv",
            remote_path=f"{REMOTE_ROOT}/artifacts/qualification/proteingym-parity-v1/variants.csv",
        )
        .add_local_file(
            REPOSITORY / "artifacts/qualification/proteingym-parity-v1/panel-manifest.json",
            remote_path=(
                f"{REMOTE_ROOT}/artifacts/qualification/proteingym-parity-v1/"
                "panel-manifest.json"
            ),
        )
        .add_local_file(
            REPOSITORY / "artifacts/qualification/proteingym-parity-v1/structure-manifest.json",
            remote_path=(
                f"{REMOTE_ROOT}/artifacts/qualification/proteingym-parity-v1/"
                "structure-manifest.json"
            ),
        )
        .add_local_dir(
            REPOSITORY / "artifacts/qualification/proteingym-parity-v1/structures",
            remote_path=(
                f"{REMOTE_ROOT}/artifacts/qualification/proteingym-parity-v1/structures"
            ),
        )
    )


base_image = CONTAINER_LOCK["base_image"]
fair_esm_image = _inputs(
    modal.Image.from_registry(base_image)
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
    ),
    "fair_esm",
)

external_image = _inputs(
    modal.Image.from_registry(base_image)
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
        "wget -q https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz "
        "-O /tmp/foldseek.tar.gz",
        "echo 'b6d590b7a0eb4bdc6d825065e8b5416397bf2822c588770508c1152f62676c79  "
        "/tmp/foldseek.tar.gz' | sha256sum -c -",
        "tar -xzf /tmp/foldseek.tar.gz -C /opt",
        "rm /tmp/foldseek.tar.gz",
    )
    .env(
        {
            "PROTEINMPNN_DIR": "/opt/ProteinMPNN",
            "TRANCEPTION_DIR": "/opt/Tranception",
            "FOLDSEEK_BINARY": "/opt/foldseek/bin/foldseek",
        }
    ),
    "external",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        item for item in root.rglob("*.py") if item.is_file() and "__pycache__" not in item.parts
    ):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _dataset_paths(dataset: str) -> tuple[Path, Path, Optional[Path], Path, Optional[Path]]:
    root = Path(REMOTE_ROOT)
    if dataset == "domainome":
        base = root / "results/confirmation/domainome-v1"
        return (
            base / "targets.csv",
            base / "variants.csv",
            base / "outcome-lock.json",
            root / "artifacts/confirmation/domainome-v1/structures/structures",
            root
            / "artifacts/confirmation/domainome-v1/structures/structure-input-manifest.json",
        )
    if dataset == "parity":
        base = root / "artifacts/qualification/proteingym-parity-v1"
        return (
            base / "targets.csv",
            base / "variants.csv",
            None,
            base / "structures",
            base / "structure-manifest.json",
        )
    raise ValueError(f"Unsupported qualification dataset: {dataset}")


def _execute(
    model_id: str,
    dataset: str,
    run_id: str,
    target_limit: Optional[int],
    target_offset: int,
    probe: bool,
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
        prediction_cache_key,
    )
    from variantshift.schemas import write_table

    root = Path(REMOTE_ROOT)
    config_path = root / "configs/model-panel-v1.json"
    qualification_path = root / "configs/qualification-v1.json"
    container_lock_path = root / "containers/qualification-lock-v1.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    allowed_runs = set(map(str, qualification["independent_runs"]))
    parity_run_id = str(qualification.get("parity_run_id", "parity-reference"))
    if not probe and run_id not in allowed_runs and run_id != parity_run_id:
        raise ValueError(f"Run id is not frozen in the qualification protocol: {run_id}")
    if probe and not run_id.startswith("checkpoint-probe-"):
        raise ValueError("Probe run identifiers must start with checkpoint-probe-")
    targets_path, variants_path, lock_path, structure_dir, structure_manifest = (
        _dataset_paths(dataset)
    )
    if lock_path is not None:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        required = str(qualification["confirmation_lock_required_state"])
        if lock.get("state") != required:
            raise RuntimeError(
                f"Qualification requires outcome lock {required}, found {lock.get('state')}"
            )
    specifications = load_model_specifications(config_path)
    specification = next(item for item in specifications if item.model_id == model_id)
    base_specification_sha256 = specification.digest()
    if dataset == "parity" and "structure" in specification.modalities:
        specification = dataclasses.replace(
            specification,
            structure_dir=str(structure_dir),
            input_manifest=str(structure_manifest),
        )
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
    cache_root = (
        Path("/qualification-cache")
        / str(qualification["protocol_id"])
        / dataset
        / run_id
    )
    model_cache_dir = cache_root / model_id
    preexisting = []
    for target in targets.itertuples(index=False):
        current = variants.loc[variants["target_id"].astype(str).eq(str(target.target_id))]
        key = prediction_cache_key(specification, pd.Series(target._asdict()), current)
        candidate = model_cache_dir / f"{key}.csv"
        if candidate.exists():
            preexisting.append(str(target.target_id))
    if preexisting:
        raise RuntimeError(
            f"Independent qualification cache is not fresh for {model_id}: {preexisting[:5]}"
        )
    # Stage target caches on the container-local filesystem. A preempted container must not
    # contaminate the persistent namespace and make an otherwise valid retry look like a cached
    # rerun. The complete cache tree is published only after every target has finished.
    local_cache_root = (
        Path("/tmp/qualification-cache")
        / str(qualification["protocol_id"])
        / dataset
        / run_id
    )
    if local_cache_root.exists():
        shutil.rmtree(local_cache_root)

    started_at = datetime.now(timezone.utc)
    started = time.time()
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    from variantshift.model_adapters import score_panel

    predictions, audit = score_panel(
        adapter,
        targets,
        variants,
        protocol_id=str(qualification["protocol_id"]),
        cache_dir=local_cache_root.parent / run_id,
    )
    elapsed = time.time() - started
    if bool(audit["cache_hit"].astype(bool).any()):
        raise RuntimeError("Qualification execution observed a cache hit")
    provenance = adapter.provenance()
    # Persist any checkpoint files fetched during adapter initialization before publishing the
    # execution receipt. Prediction independence does not require redownloading identical weights.
    model_cache.commit()
    expected_checkpoint = specification.checkpoint_sha256
    if not probe and expected_checkpoint != provenance.get("checkpoint_sha256"):
        raise RuntimeError(
            f"Checkpoint hash mismatch for {model_id}: expected {expected_checkpoint}, "
            f"found {provenance.get('checkpoint_sha256')}"
        )
    expected_container = _container_sha256(os.environ["VARIANTSHIFT_CONTAINER_KIND"])
    if provenance.get("container_sha256") != expected_container:
        raise RuntimeError(f"Container provenance mismatch for {model_id}")
    local_model_cache = local_cache_root / model_id
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    for cache_artifact in local_model_cache.iterdir():
        destination_cache = model_cache_dir / cache_artifact.name
        if destination_cache.exists():
            raise RuntimeError(
                f"Qualification cache key became non-fresh for {model_id}: "
                f"{cache_artifact.name}"
            )
        shutil.copy2(cache_artifact, destination_cache)
    qualification_cache.commit()

    temporary = Path("/tmp") / f"qualification-{dataset}-{run_id}-{model_id}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    prediction_path = write_table(predictions, temporary / "predictions.csv.gz")
    audit_path = write_table(audit, temporary / "prediction-audit.csv.gz")
    provenance_path = temporary / "model-provenance.json"
    provenance_path.write_text(
        json.dumps({model_id: provenance}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    passing = sorted(
        audit.loc[audit["status"].eq("ok") & audit["coverage"].ge(0.95), "target_id"]
        .astype(str)
        .tolist()
    )
    failures = audit.loc[audit["status"].ne("ok"), ["target_id", "status", "error"]]
    manifest = {
        "schema_version": 1,
        "protocol_id": str(qualification["protocol_id"]),
        "confirmation_protocol_id": str(qualification["confirmation_protocol_id"]),
        "dataset": dataset,
        "run_id": run_id,
        "probe": probe,
        "model_id": model_id,
        "family": specification.family,
        "execution_status": "complete",
        "qualification_status": "probe" if probe else "candidate",
        "target_count_requested": int(targets["target_id"].nunique()),
        "target_offset": target_offset,
        "targets_at_95pct_coverage": len(passing),
        "prediction_rows": len(predictions),
        "variant_rows_requested": len(variants),
        "substitution_coverage": len(predictions) / max(len(variants), 1),
        "cache_namespace": str(cache_root),
        "cache_namespace_fresh": True,
        "cache_hit_count": 0,
        "elapsed_seconds": elapsed,
        "started_at": started_at.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "hardware": {
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_version": torch.version.cuda,
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
            ),
            "platform": platform.platform(),
        },
        "failures": failures.to_dict(orient="records"),
        "inputs": {
            "model_config_sha256": _sha256(config_path),
            "qualification_config_sha256": _sha256(qualification_path),
            "container_lock_sha256": _sha256(container_lock_path),
            "container_sha256": expected_container,
            "targets_sha256": _sha256(targets_path),
            "variants_sha256": _sha256(variants_path),
            "structure_manifest_sha256": (
                _sha256(structure_manifest)
                if structure_manifest is not None and structure_manifest.is_file()
                else None
            ),
            "outcome_lock_sha256": _sha256(lock_path) if lock_path is not None else None,
            "source_tree_sha256": _tree_sha256(root / "src"),
            "runner_sha256": _sha256(root / "workflow/modal_qualification.py"),
            "base_specification_sha256": base_specification_sha256,
            "effective_specification_sha256": specification.digest(),
        },
        "artifacts": {
            prediction_path.name: _sha256(prediction_path),
            audit_path.name: _sha256(audit_path),
            provenance_path.name: _sha256(provenance_path),
        },
        "confirmation_outcomes_accessed": False,
    }
    manifest_path = temporary / "execution-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    last_offset = target_offset + len(targets) - 1
    destination = (
        Path("/qualification/runs")
        / dataset
        / run_id
        / model_id
        / "shards"
        / f"{target_offset:04d}-{last_offset:04d}"
    )
    if destination.exists():
        raise RuntimeError(f"Immutable qualification shard already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(temporary, destination)
    qualification_volume.commit()
    return manifest


@app.function(
    image=fair_esm_image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={
        "/qualification": qualification_volume,
        "/qualification-cache": qualification_cache,
        "/model-cache": model_cache,
    },
)
def execute_fair_esm(
    model_id: str,
    dataset: str,
    run_id: str,
    target_limit: Optional[int] = None,
    target_offset: int = 0,
    probe: bool = False,
) -> dict[str, object]:
    allowed = {
        "esm1v_ensemble",
        "esm2_8m",
        "esm2_35m",
        "esm2_150m",
        "esm2_650m",
        "esm_if1",
    }
    if model_id not in allowed:
        raise ValueError(f"Unsupported fair-esm qualification model: {model_id}")
    return _execute(model_id, dataset, run_id, target_limit, target_offset, probe)


@app.function(
    image=external_image,
    gpu="L4",
    cpu=8,
    memory=49152,
    timeout=24 * 60 * 60,
    volumes={
        "/qualification": qualification_volume,
        "/qualification-cache": qualification_cache,
        "/model-cache": model_cache,
    },
)
def execute_external(
    model_id: str,
    dataset: str,
    run_id: str,
    target_limit: Optional[int] = None,
    target_offset: int = 0,
    probe: bool = False,
) -> dict[str, object]:
    allowed = {"proteinmpnn", "saprot_35m", "tranception_l_no_retrieval"}
    if model_id not in allowed:
        raise ValueError(f"Unsupported external qualification model: {model_id}")
    return _execute(model_id, dataset, run_id, target_limit, target_offset, probe)


@app.function(
    image=external_image,
    cpu=4,
    memory=16384,
    timeout=60 * 60,
    volumes={"/qualification": qualification_volume},
)
def merge_shards(
    model_id: str,
    dataset: str,
    run_id: str,
    expected_targets: int,
) -> dict[str, object]:
    import os
    import shutil

    import pandas as pd

    os.chdir(REMOTE_ROOT)
    from variantshift.schemas import PREDICTION_SCHEMA, write_table

    source = Path("/qualification/runs") / dataset / run_id / model_id / "shards"
    shard_directories = sorted(path for path in source.iterdir() if path.is_dir())
    if not shard_directories:
        raise RuntimeError(f"No qualification shards found for {dataset}/{run_id}/{model_id}")
    prediction_frames = [pd.read_csv(path / "predictions.csv.gz") for path in shard_directories]
    audit_frames = [pd.read_csv(path / "prediction-audit.csv.gz") for path in shard_directories]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    audits = pd.concat(audit_frames, ignore_index=True)
    if predictions.duplicated(PREDICTION_SCHEMA.unique).any():
        raise RuntimeError("Qualification shards contain duplicate prediction keys")
    if audits["target_id"].astype(str).duplicated().any():
        raise RuntimeError("Qualification shards contain duplicate target audits")
    if audits["target_id"].nunique() != expected_targets:
        raise RuntimeError(
            f"Expected {expected_targets} targets for {model_id}, found "
            f"{audits['target_id'].nunique()}"
        )
    if audits["cache_hit"].astype(bool).any():
        raise RuntimeError("Qualification merge found a cache hit")
    PREDICTION_SCHEMA.validate(predictions)
    manifests = [
        json.loads((path / "execution-manifest.json").read_text(encoding="utf-8"))
        for path in shard_directories
    ]
    provenances = [
        json.loads((path / "model-provenance.json").read_text(encoding="utf-8"))[model_id]
        for path in shard_directories
    ]
    stable_provenance = {
        (
            item.get("specification_sha256"),
            item.get("checkpoint_sha256"),
            item.get("container_sha256"),
            item.get("input_manifest_sha256"),
        )
        for item in provenances
    }
    if len(stable_provenance) != 1:
        raise RuntimeError("Qualification shard provenance is inconsistent")
    stable_inputs = {
        json.dumps(item["inputs"], sort_keys=True, separators=(",", ":"))
        for item in manifests
    }
    if len(stable_inputs) != 1:
        raise RuntimeError("Qualification shard inputs are inconsistent")

    temporary = Path("/tmp") / f"qualification-merge-{dataset}-{run_id}-{model_id}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    prediction_path = write_table(predictions, temporary / "predictions.csv.gz")
    audit_path = write_table(audits, temporary / "prediction-audit.csv.gz")
    provenance_path = temporary / "model-provenance.json"
    provenance_path.write_text(
        json.dumps({model_id: provenances[0]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    failures = audits.loc[audits["status"].ne("ok"), ["target_id", "status", "error"]]
    manifest = {
        "schema_version": 1,
        "protocol_id": manifests[0]["protocol_id"],
        "confirmation_protocol_id": manifests[0]["confirmation_protocol_id"],
        "dataset": dataset,
        "run_id": run_id,
        "probe": bool(manifests[0]["probe"]),
        "model_id": model_id,
        "family": manifests[0]["family"],
        "execution_status": "complete",
        "qualification_status": manifests[0]["qualification_status"],
        "target_count_requested": expected_targets,
        "targets_at_95pct_coverage": int(
            (audits["status"].eq("ok") & audits["coverage"].ge(0.95)).sum()
        ),
        "prediction_rows": len(predictions),
        "variant_rows_requested": int(sum(item["variant_rows_requested"] for item in manifests)),
        "substitution_coverage": len(predictions)
        / max(int(sum(item["variant_rows_requested"] for item in manifests)), 1),
        "cache_namespaces": [item["cache_namespace"] for item in manifests],
        "cache_namespace_fresh": True,
        "cache_hit_count": 0,
        "elapsed_gpu_seconds_sum": sum(float(item["elapsed_seconds"]) for item in manifests),
        "started_at": min(item["started_at"] for item in manifests),
        "finished_at": max(item["finished_at"] for item in manifests),
        "hardware": [item["hardware"] for item in manifests],
        "failures": failures.to_dict(orient="records"),
        "execution_shards": [path.name for path in shard_directories],
        "inputs": manifests[0]["inputs"],
        "artifacts": {
            prediction_path.name: _sha256(prediction_path),
            audit_path.name: _sha256(audit_path),
            provenance_path.name: _sha256(provenance_path),
        },
        "confirmation_outcomes_accessed": False,
    }
    manifest_path = temporary / "execution-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    destination = Path("/qualification/runs") / dataset / run_id / model_id
    for artifact in temporary.iterdir():
        final = destination / artifact.name
        if final.exists():
            raise RuntimeError(f"Immutable merged artifact already exists: {final}")
        shutil.copy2(artifact, final)
    qualification_volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    model_id: str,
    dataset: str = "domainome",
    run_id: str = "qualification-a",
    target_limit: Optional[int] = None,
    target_offset: int = 0,
    probe: bool = False,
    merge: bool = False,
    expected_targets: int = 426,
    shard_count: int = 1,
) -> None:
    if merge:
        result = merge_shards.remote(model_id, dataset, run_id, expected_targets)
    else:
        fair_models = {
            "esm1v_ensemble",
            "esm2_8m",
            "esm2_35m",
            "esm2_150m",
            "esm2_650m",
            "esm_if1",
        }
        function = execute_fair_esm if model_id in fair_models else execute_external
        if shard_count < 1:
            raise ValueError("shard_count must be positive")
        if shard_count > 1:
            if target_limit is not None or target_offset != 0:
                raise ValueError(
                    "target_limit and target_offset cannot be combined with shard_count"
                )
            shard_size = (expected_targets + shard_count - 1) // shard_count
            arguments = [
                (
                    model_id,
                    dataset,
                    run_id,
                    min(shard_size, expected_targets - offset),
                    offset,
                    probe,
                )
                for offset in range(0, expected_targets, shard_size)
            ]
            shard_results = list(
                function.starmap(
                    arguments,
                    order_outputs=True,
                    return_exceptions=True,
                )
            )
            failures = [repr(item) for item in shard_results if isinstance(item, Exception)]
            if failures:
                raise RuntimeError(
                    "Qualification shards failed after all scheduled work completed: "
                    + "; ".join(failures)
                )
            result = shard_results
        else:
            result = function.remote(
                model_id,
                dataset,
                run_id,
                target_limit,
                target_offset,
                probe,
            )
    print(json.dumps(result, indent=2, sort_keys=True))
