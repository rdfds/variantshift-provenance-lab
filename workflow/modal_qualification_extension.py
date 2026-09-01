"""Fail-closed Modal runner for the outcome-free model-panel extension."""

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
LOCAL_REPOSITORY = Path(__file__).resolve().parents[1]
REPOSITORY = (
    LOCAL_REPOSITORY
    if (LOCAL_REPOSITORY / "containers/qualification-extension-lock-v3.json").is_file()
    else Path(REMOTE_ROOT)
)
LOCK_PATH = REPOSITORY / "containers/qualification-extension-lock-v3.json"
CONTAINER_LOCK = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
QUALIFICATION_VOLUME = "variantshift-qualification-extension-v3"
CACHE_VOLUME = "variantshift-qualification-extension-cache-v3"
MODEL_CACHE_VOLUME = "variantshift-model-cache-v2"

qualification_volume = modal.Volume.from_name(QUALIFICATION_VOLUME, create_if_missing=True)
qualification_cache = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)
model_cache = modal.Volume.from_name(MODEL_CACHE_VOLUME, create_if_missing=True)
app = modal.App("variantshift-model-qualification-extension")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _container_sha256() -> str:
    return _canonical_sha256(
        {
            "base_image": CONTAINER_LOCK["base_image"],
            "image": "extension",
            **CONTAINER_LOCK["images"]["extension"],
        }
    )


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


def _add_inputs(image: modal.Image) -> modal.Image:
    parity = REPOSITORY / "artifacts/qualification/proteingym-parity-extension-v3"
    return (
        image.env(
            {
                "PYTHONPATH": f"{REMOTE_ROOT}/src",
                "PYTHONHASHSEED": "0",
                "HF_HOME": "/model-cache/huggingface",
                "TORCH_HOME": "/model-cache/torch",
                "CARP_DIR": "/opt/protein-sequence-models",
                "VESPAG_DIR": "/opt/VespaG",
                "VARIANTSHIFT_CONTAINER_SHA256": _container_sha256(),
                "VARIANTSHIFT_CONTAINER_KIND": "extension",
            }
        )
        .add_local_dir(REPOSITORY / "src", remote_path=f"{REMOTE_ROOT}/src")
        .add_local_file(
            REPOSITORY / "workflow/modal_qualification_extension.py",
            remote_path=f"{REMOTE_ROOT}/workflow/modal_qualification_extension.py",
        )
        .add_local_file(
            REPOSITORY / "configs/model-panel-extension-v3.json",
            remote_path=f"{REMOTE_ROOT}/configs/model-panel-extension-v3.json",
        )
        .add_local_file(
            REPOSITORY / "configs/qualification-extension-v3.json",
            remote_path=f"{REMOTE_ROOT}/configs/qualification-extension-v3.json",
        )
        .add_local_file(
            LOCK_PATH,
            remote_path=f"{REMOTE_ROOT}/containers/qualification-extension-lock-v3.json",
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
            parity / "targets.csv",
            remote_path=(
                f"{REMOTE_ROOT}/artifacts/qualification/"
                "proteingym-parity-extension-v3/targets.csv"
            ),
        )
        .add_local_file(
            parity / "variants.csv",
            remote_path=(
                f"{REMOTE_ROOT}/artifacts/qualification/"
                "proteingym-parity-extension-v3/variants.csv"
            ),
        )
        .add_local_file(
            parity / "structure-manifest.json",
            remote_path=(
                f"{REMOTE_ROOT}/artifacts/qualification/"
                "proteingym-parity-extension-v3/structure-manifest.json"
            ),
        )
        .add_local_dir(
            parity / "structures",
            remote_path=(
                f"{REMOTE_ROOT}/artifacts/qualification/"
                "proteingym-parity-extension-v3/structures"
            ),
        )
    )


extension_image = _add_inputs(
    modal.Image.from_registry(CONTAINER_LOCK["base_image"])
    .apt_install("git", "ca-certificates")
    .pip_install(
        "beartype==0.20.2",
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scipy==1.14.1",
        "scikit-learn==1.6.1",
        "transformers==4.38.2",
        "huggingface-hub==0.24.7",
        "jaxtyping==0.2.37",
    )
    .run_commands(
        "git clone https://github.com/microsoft/protein-sequence-models.git "
        "/opt/protein-sequence-models",
        "git -C /opt/protein-sequence-models checkout "
        "af695772c4a1c056d930c95ec7e6428aa042f5cd",
        "git clone https://github.com/JSchlensok/VespaG.git /opt/VespaG",
        "git -C /opt/VespaG checkout 3d4758252dbd423249e694d6f7d195c707f72a92",
        "test \"$(sha256sum /opt/VespaG/model_weights/v2/esm2.pt | cut -d' ' -f1)\" = "
        "cf30b714bed4466930fc66fcf0f1594a4ba5192e3885e60a743c808dd9f66dd2",
    )
)


@app.function(
    image=extension_image,
    cpu=2,
    memory=4096,
    timeout=60 * 60,
    volumes={"/model-cache": model_cache},
)
def download_carp_checkpoint() -> dict[str, object]:
    """Populate and verify the persistent CARP checkpoint without inference retries."""
    import hashlib
    import urllib.request

    destination = Path("/model-cache/torch/hub/checkpoints/carp_640M.pt")
    expected_size = 2572014451
    expected_md5 = "2601f6b9971fe5ea1733067eb87a3398"
    url = "https://zenodo.org/api/records/6564798/files/carp_640M.pt/content"

    def md5_file(path: Path) -> str:
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    if destination.is_file() and destination.stat().st_size == expected_size:
        digest = md5_file(destination)
        if digest == expected_md5:
            return {"status": "already_verified", "size_bytes": expected_size, "md5": digest}
    partial = destination.with_suffix(".pt.partial")
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_error = ""
    for attempt in range(12):
        offset = partial.stat().st_size if partial.is_file() else 0
        if offset == expected_size:
            break
        if offset > expected_size:
            raise RuntimeError(f"CARP partial checkpoint exceeds expected size: {offset}")
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"})
        try:
            response = urllib.request.urlopen(request, timeout=120)
            if offset and response.status != 206:
                offset = 0
            mode = "ab" if offset else "wb"
            with response, partial.open(mode) as handle:
                while block := response.read(8 * 1024 * 1024):
                    handle.write(block)
        except Exception as exception:  # noqa: BLE001 - bounded resumable transfer
            last_error = f"{type(exception).__name__}: {exception}"
        model_cache.commit()
    if partial.stat().st_size != expected_size:
        raise RuntimeError(
            f"CARP checkpoint download is incomplete after 12 attempts: "
            f"{partial.stat().st_size}/{expected_size}; last error: {last_error}"
        )
    digest = md5_file(partial)
    if digest != expected_md5:
        raise RuntimeError(f"CARP checkpoint MD5 mismatch: {digest}")
    partial.replace(destination)
    model_cache.commit()
    return {"status": "downloaded", "size_bytes": expected_size, "md5": digest}


def _dataset_paths(dataset: str) -> tuple[Path, Path, Optional[Path], Path, Path]:
    root = Path(REMOTE_ROOT)
    if dataset == "domainome":
        base = root / "results/confirmation/domainome-v1"
        structure = root / "artifacts/confirmation/domainome-v1/structures"
        return (
            base / "targets.csv",
            base / "variants.csv",
            base / "outcome-lock.json",
            structure / "structures",
            structure / "structure-input-manifest.json",
        )
    if dataset == "parity":
        base = root / "artifacts/qualification/proteingym-parity-extension-v3"
        return (
            base / "targets.csv",
            base / "variants.csv",
            None,
            base / "structures",
            base / "structure-manifest.json",
        )
    raise ValueError(f"Unsupported extension dataset: {dataset}")


@app.function(
    image=extension_image,
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
def execute(
    model_id: str,
    dataset: str,
    run_id: str,
    target_limit: Optional[int] = None,
    target_offset: int = 0,
    probe: bool = False,
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
        score_panel,
    )
    from variantshift.schemas import write_table

    if model_id not in {"carp_640m", "vespag"}:
        raise ValueError(f"Unsupported extension model: {model_id}")
    root = Path(REMOTE_ROOT)
    model_config_path = root / "configs/model-panel-extension-v3.json"
    qualification_path = root / "configs/qualification-extension-v3.json"
    container_lock_path = root / "containers/qualification-extension-lock-v3.json"
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    allowed_runs = set(map(str, qualification["independent_runs"]))
    parity_run = str(qualification["parity_run_id"])
    if not probe and run_id not in allowed_runs | {parity_run}:
        raise ValueError(f"Run id is not frozen in the extension protocol: {run_id}")
    if probe and not run_id.startswith("checkpoint-probe-"):
        raise ValueError("Probe run identifiers must start with checkpoint-probe-")
    targets_path, variants_path, lock_path, structure_dir, structure_manifest = _dataset_paths(
        dataset
    )
    if lock_path is not None:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if lock.get("state") != qualification["confirmation_lock_required_state"]:
            raise RuntimeError("Confirmation outcome lock is not in targets_frozen state")
        if lock.get("registration") is not None or lock.get("reveal") is not None:
            raise RuntimeError("Extension qualification must precede registration and reveal")
    specification = next(
        item
        for item in load_model_specifications(model_config_path)
        if item.model_id == model_id
    )
    base_specification_sha256 = specification.digest()
    if "structure" in specification.modalities:
        specification = dataclasses.replace(
            specification,
            structure_dir=str(structure_dir),
            input_manifest=str(structure_manifest),
        )
    targets = pd.read_csv(targets_path).sort_values("target_id").reset_index(drop=True)
    if target_offset < 0 or target_offset >= len(targets):
        raise ValueError("target_offset is outside the frozen panel")
    targets = targets.iloc[target_offset:]
    if target_limit is not None:
        if target_limit < 1:
            raise ValueError("target_limit must be positive")
        targets = targets.head(target_limit)
    target_ids = set(targets["target_id"].astype(str))
    variants = pd.read_csv(variants_path)
    variants = variants.loc[variants["target_id"].astype(str).isin(target_ids)].copy()
    adapter = adapter_from_specification(specification)
    cache_root = (
        Path("/qualification-cache")
        / qualification["protocol_id"]
        / dataset
        / run_id
    )
    model_cache_root = cache_root / model_id
    for target in targets.itertuples(index=False):
        current = variants.loc[variants["target_id"].astype(str).eq(str(target.target_id))]
        key = prediction_cache_key(specification, pd.Series(target._asdict()), current)
        if (model_cache_root / f"{key}.csv").exists():
            raise RuntimeError(f"Independent extension cache is not fresh for {target.target_id}")
    local_cache = Path("/tmp/qualification-extension-cache") / dataset / run_id
    if local_cache.exists():
        shutil.rmtree(local_cache)
    started_at = datetime.now(timezone.utc)
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    predictions, audit = score_panel(
        adapter,
        targets,
        variants,
        protocol_id=qualification["protocol_id"],
        cache_dir=local_cache,
    )
    if audit["cache_hit"].astype(bool).any():
        raise RuntimeError("Extension qualification observed a cache hit")
    provenance = adapter.provenance()
    model_cache.commit()
    expected_checkpoint = specification.checkpoint_sha256
    if not probe and (
        not expected_checkpoint or expected_checkpoint != provenance.get("checkpoint_sha256")
    ):
        raise RuntimeError(
            f"Checkpoint hash mismatch for {model_id}: expected {expected_checkpoint}, "
            f"found {provenance.get('checkpoint_sha256')}"
        )
    expected_container = _container_sha256()
    if provenance.get("container_sha256") != expected_container:
        raise RuntimeError("Extension container provenance mismatch")
    local_model_cache = local_cache / model_id
    model_cache_root.mkdir(parents=True, exist_ok=True)
    for path in local_model_cache.iterdir():
        destination = model_cache_root / path.name
        if destination.exists():
            raise RuntimeError(f"Extension cache key became non-fresh: {path.name}")
        shutil.copy2(path, destination)
    qualification_cache.commit()

    temporary = Path("/tmp") / f"qualification-extension-{dataset}-{run_id}-{model_id}"
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
    failures = audit.loc[audit["status"].ne("ok"), ["target_id", "status", "error"]]
    passing = audit["status"].eq("ok") & audit["coverage"].ge(0.95)
    manifest = {
        "schema_version": 1,
        "protocol_id": qualification["protocol_id"],
        "confirmation_protocol_id": qualification["confirmation_protocol_id"],
        "dataset": dataset,
        "run_id": run_id,
        "probe": probe,
        "model_id": model_id,
        "family": specification.family,
        "execution_status": "complete",
        "qualification_status": "probe" if probe else "candidate",
        "target_count_requested": int(targets["target_id"].nunique()),
        "targets_at_95pct_coverage": int(passing.sum()),
        "prediction_rows": len(predictions),
        "variant_rows_requested": len(variants),
        "substitution_coverage": len(predictions) / max(len(variants), 1),
        "cache_namespace": str(cache_root),
        "cache_namespace_fresh": True,
        "cache_hit_count": 0,
        "elapsed_seconds": time.time() - started,
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
            "model_config_sha256": _sha256(model_config_path),
            "qualification_config_sha256": _sha256(qualification_path),
            "container_lock_sha256": _sha256(container_lock_path),
            "container_sha256": expected_container,
            "targets_sha256": _sha256(targets_path),
            "variants_sha256": _sha256(variants_path),
            "structure_manifest_sha256": _sha256(structure_manifest),
            "outcome_lock_sha256": _sha256(lock_path) if lock_path is not None else None,
            "source_tree_sha256": _tree_sha256(root / "src"),
            "runner_sha256": _sha256(root / "workflow/modal_qualification_extension.py"),
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
    destination = Path("/qualification/runs") / dataset / run_id / model_id
    if destination.exists():
        raise RuntimeError(f"Immutable extension execution already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(temporary, destination)
    qualification_volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    model_id: str = "carp_640m",
    dataset: str = "domainome",
    run_id: str = "qualification-extension-a",
    target_limit: Optional[int] = None,
    target_offset: int = 0,
    probe: bool = False,
    download_carp: bool = False,
) -> None:
    if download_carp:
        result = download_carp_checkpoint.remote()
    else:
        result = execute.remote(model_id, dataset, run_id, target_limit, target_offset, probe)
    print(json.dumps(result, indent=2, sort_keys=True))
