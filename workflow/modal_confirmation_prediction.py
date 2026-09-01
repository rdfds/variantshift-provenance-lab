"""Resumable outcome-blind Modal scoring for the frozen external panels."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import modal

REMOTE_ROOT = "/root/variantshift"
LOCAL_REPOSITORY = Path(__file__).resolve().parents[1]
REPOSITORY = (
    LOCAL_REPOSITORY
    if (LOCAL_REPOSITORY / "containers/qualification-lock-v1.json").is_file()
    else Path(REMOTE_ROOT)
)
FINAL_CONFIG = REPOSITORY / "configs/model-panel-final-v1.json"
FINAL_RECEIPT = REPOSITORY / "results/model-qualification-final-v1/qualification-summary.json"
PREDICTION_VOLUME = "variantshift-confirmation-predictions-v1"
PREDICTION_CACHE_VOLUME = "variantshift-confirmation-prediction-cache-v1"
RUN_ROOT = Path("/predictions/final-v2")

prediction_volume = modal.Volume.from_name(PREDICTION_VOLUME, create_if_missing=True)
prediction_cache = modal.Volume.from_name(PREDICTION_CACHE_VOLUME, create_if_missing=True)
model_cache_v1 = modal.Volume.from_name("variantshift-model-cache-v1", create_if_missing=True)
model_cache_v2 = modal.Volume.from_name("variantshift-model-cache-v2", create_if_missing=True)
app = modal.App("variantshift-confirmation-prediction-freeze")


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


def _container_sha256(lock_name: str, image_name: str) -> str:
    qualified = {
        "fair_esm": "5e0045fdeaa80c4b408b9511dde626afa9f1fe76906707640a3aa66c3b1ec3ef",
        "external": "20775869789db6087f13ec5634692b08657fb833c210b3088dae1236be960efc",
        "extension": "11373f28857fab398f3f0692f2e05c7523a48abad049980c2b8920f943615077",
    }
    lock_path = REPOSITORY / lock_name
    if not lock_path.is_file():
        return qualified[image_name]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    return _canonical_sha256(
        {"base_image": lock["base_image"], "image": image_name, **lock["images"][image_name]}
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


def _common_inputs(image: modal.Image, container_sha256: str, container_kind: str) -> modal.Image:
    return (
        image.env(
            {
                "PYTHONPATH": f"{REMOTE_ROOT}/src",
                "PYTHONHASHSEED": "0",
                "HF_HOME": "/model-cache/huggingface",
                "TORCH_HOME": "/model-cache/torch",
                "VARIANTSHIFT_CONTAINER_SHA256": container_sha256,
                "VARIANTSHIFT_CONTAINER_KIND": container_kind,
            }
        )
        .add_local_dir(REPOSITORY / "src", remote_path=f"{REMOTE_ROOT}/src")
        .add_local_file(
            REPOSITORY / "workflow/modal_confirmation_prediction.py",
            remote_path=f"{REMOTE_ROOT}/workflow/modal_confirmation_prediction.py",
        )
        .add_local_file(FINAL_CONFIG, remote_path=f"{REMOTE_ROOT}/configs/model-panel-final-v1.json")
        .add_local_file(
            FINAL_RECEIPT,
            remote_path=(
                f"{REMOTE_ROOT}/results/model-qualification-final-v1/qualification-summary.json"
            ),
        )
        .add_local_file(
            REPOSITORY / "containers/qualification-lock-v1.json",
            remote_path=f"{REMOTE_ROOT}/containers/qualification-lock-v1.json",
        )
        .add_local_file(
            REPOSITORY / "containers/qualification-extension-lock-v3.json",
            remote_path=(
                f"{REMOTE_ROOT}/containers/qualification-extension-lock-v3.json"
            ),
        )
        .add_local_file(
            REPOSITORY / "protocols/mavedb-complement-v1/frozen/targets.csv",
            remote_path=f"{REMOTE_ROOT}/protocols/mavedb-complement-v1/frozen/targets.csv",
        )
        .add_local_file(
            REPOSITORY / "protocols/mavedb-complement-v1/frozen/variants.csv",
            remote_path=f"{REMOTE_ROOT}/protocols/mavedb-complement-v1/frozen/variants.csv",
        )
        .add_local_file(
            REPOSITORY / "protocols/mavedb-complement-v1/frozen/outcome-lock.json",
            remote_path=(
                f"{REMOTE_ROOT}/protocols/mavedb-complement-v1/frozen/outcome-lock.json"
            ),
        )
        .add_local_dir(
            REPOSITORY / "artifacts/confirmation/mavedb-complement-v1",
            remote_path=f"{REMOTE_ROOT}/artifacts/confirmation/mavedb-complement-v1",
        )
        .add_local_file(
            REPOSITORY / "protocols/venusmuthub-v1/frozen/targets.csv",
            remote_path=f"{REMOTE_ROOT}/protocols/venusmuthub-v1/frozen/targets.csv",
        )
        .add_local_file(
            REPOSITORY / "protocols/venusmuthub-v1/frozen/variants.csv.gz",
            remote_path=f"{REMOTE_ROOT}/protocols/venusmuthub-v1/frozen/variants.csv.gz",
        )
        .add_local_file(
            REPOSITORY / "protocols/venusmuthub-v1/frozen/outcome-lock.json",
            remote_path=f"{REMOTE_ROOT}/protocols/venusmuthub-v1/frozen/outcome-lock.json",
        )
        .add_local_dir(
            REPOSITORY / "artifacts/confirmation/venusmuthub-v1",
            remote_path=f"{REMOTE_ROOT}/artifacts/confirmation/venusmuthub-v1",
        )
    )


_BASE_IMAGE = (
    "pytorch/pytorch@sha256:"
    "923f687790bec78081c357e71dcd5dcef80b0cc00f6c34484902a5e83362c854"
)
lock_v1_path = REPOSITORY / "containers/qualification-lock-v1.json"
lock_v3_path = REPOSITORY / "containers/qualification-extension-lock-v3.json"
lock_v1 = (
    json.loads(lock_v1_path.read_text(encoding="utf-8"))
    if lock_v1_path.is_file()
    else {"base_image": _BASE_IMAGE}
)
lock_v3 = (
    json.loads(lock_v3_path.read_text(encoding="utf-8"))
    if lock_v3_path.is_file()
    else {"base_image": _BASE_IMAGE}
)
base_v1 = lock_v1["base_image"]
fair_image = _common_inputs(
    modal.Image.from_registry(base_v1)
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
    _container_sha256("containers/qualification-lock-v1.json", "fair_esm"),
    "fair_esm",
)
external_image = _common_inputs(
    modal.Image.from_registry(base_v1)
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
        "git clone https://github.com/OATML-Markslab/Tranception.git /opt/Tranception",
        "git -C /opt/Tranception checkout 2ddf40e1db9d2d180d1b5fc9d1b39ad5b04fbb6d",
        "git clone https://github.com/westlake-repl/SaProt.git /opt/SaProt",
        "git -C /opt/SaProt checkout e91e4858b55944523f1f8d385f7b96a0d3d34c1d",
        "wget -q https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz -O /tmp/foldseek.tar.gz",
        "echo 'b6d590b7a0eb4bdc6d825065e8b5416397bf2822c588770508c1152f62676c79  "
        "/tmp/foldseek.tar.gz' | sha256sum -c -",
        "tar -xzf /tmp/foldseek.tar.gz -C /opt",
        "rm /tmp/foldseek.tar.gz",
    )
    .env(
        {
            "TRANCEPTION_DIR": "/opt/Tranception",
            "FOLDSEEK_BINARY": "/opt/foldseek/bin/foldseek",
        }
    ),
    _container_sha256("containers/qualification-lock-v1.json", "external"),
    "external",
)
extension_image = _common_inputs(
    modal.Image.from_registry(lock_v3["base_image"])
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
    .env({"CARP_DIR": "/opt/protein-sequence-models", "VESPAG_DIR": "/opt/VespaG"}),
    _container_sha256("containers/qualification-extension-lock-v3.json", "extension"),
    "extension",
)


def _panel_paths(
    panel: str, *, root: Path | None = None
) -> tuple[Path, Path, Path, Path, Path, str]:
    root = Path(REMOTE_ROOT) if root is None else Path(root)
    if panel == "mavedb-complement-v1":
        base = root / "protocols/mavedb-complement-v1/frozen"
        structures = root / "artifacts/confirmation/mavedb-complement-v1"
        return (
            base / "targets.csv",
            base / "variants.csv",
            base / "outcome-lock.json",
            structures / "structures",
            structures / "structure-input-manifest.json",
            "variantshift-mavedb-complement-confirmation-v1",
        )
    if panel == "venusmuthub-v1":
        base = root / "protocols/venusmuthub-v1/frozen"
        structures = root / "artifacts/confirmation/venusmuthub-v1"
        return (
            base / "targets.csv",
            base / "variants.csv.gz",
            base / "outcome-lock.json",
            structures / "structures",
            structures / "structure-input-manifest.json",
            "variantshift-venusmuthub-confirmation-v1",
        )
    raise ValueError(f"Unsupported confirmation panel: {panel}")


def _execute(
    model_id: str,
    panel: str,
    target_offset: int,
    target_limit: int,
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

    final_config_path = Path(REMOTE_ROOT) / "configs/model-panel-final-v1.json"
    receipt_path = (
        Path(REMOTE_ROOT)
        / "results/model-qualification-final-v1/qualification-summary.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("qualification_status") != "passed":
        raise RuntimeError("Final model panel is not qualified")
    qualified = {
        str(item.model_id): item for item in load_model_specifications(final_config_path)
    }
    if model_id not in qualified:
        raise ValueError(f"Model is absent from the final qualified panel: {model_id}")
    specification = qualified[model_id]
    base_specification_sha256 = specification.digest()
    targets_path, variants_path, lock_path, structure_dir, structure_manifest, protocol_id = (
        _panel_paths(panel)
    )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("state") != "targets_frozen":
        raise RuntimeError("External predictions require a targets_frozen outcome lock")
    if lock.get("registration") is not None or lock.get("reveal") is not None:
        raise RuntimeError("External prediction scoring must precede registration and reveal")
    if "structure" in specification.modalities:
        specification = dataclasses.replace(
            specification,
            structure_dir=str(structure_dir),
            input_manifest=str(structure_manifest),
        )
    targets_all = pd.read_csv(targets_path).sort_values("target_id").reset_index(drop=True)
    if target_offset < 0 or target_offset >= len(targets_all):
        raise ValueError("Target offset is outside the frozen panel")
    targets = targets_all.iloc[target_offset : target_offset + target_limit].copy()
    target_ids = set(targets["target_id"].astype(str))
    variants = pd.read_csv(variants_path)
    variants = variants.loc[variants["target_id"].astype(str).isin(target_ids)].copy()
    final_offset = target_offset + len(targets) - 1
    destination = (
        RUN_ROOT
        / panel
        / model_id
        / "shards"
        / f"{target_offset:04d}-{final_offset:04d}"
    )
    input_hashes = {
        "final_model_config_sha256": _sha256(final_config_path),
        "final_qualification_receipt_sha256": _sha256(receipt_path),
        "targets_sha256": _sha256(targets_path),
        "variants_sha256": _sha256(variants_path),
        "outcome_lock_sha256": _sha256(lock_path),
        "structure_manifest_sha256": _sha256(structure_manifest),
        "source_tree_sha256": _tree_sha256(Path(REMOTE_ROOT) / "src"),
        "runner_sha256": _sha256(
            Path(REMOTE_ROOT) / "workflow/modal_confirmation_prediction.py"
        ),
        "base_specification_sha256": base_specification_sha256,
        "effective_specification_sha256": specification.digest(),
    }
    prediction_volume.reload()
    if destination.exists():
        existing_manifest_path = destination / "execution-manifest.json"
        if not existing_manifest_path.is_file():
            raise RuntimeError(f"Incomplete immutable prediction shard: {destination}")
        existing = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        expected_identity = {
            "panel_id": panel,
            "model_id": model_id,
            "target_offset": target_offset,
            "target_count_requested": len(targets),
            "variant_rows_requested": len(variants),
            "inputs": input_hashes,
        }
        for key, value in expected_identity.items():
            if existing.get(key) != value:
                raise RuntimeError(
                    f"Existing immutable shard has mismatched {key}: {destination}"
                )
        for name, digest in existing.get("artifacts", {}).items():
            artifact = destination / name
            if not artifact.is_file() or _sha256(artifact) != digest:
                raise RuntimeError(f"Existing shard artifact failed hashing: {artifact}")
        return existing
    prediction_cache.reload()
    cache_dir = Path("/prediction-cache") / protocol_id / model_id
    started_at = datetime.now(timezone.utc)
    started = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    adapter = adapter_from_specification(specification)
    predictions, audit = score_panel(
        adapter,
        targets,
        variants,
        protocol_id=protocol_id,
        cache_dir=cache_dir.parent,
    )
    provenance = adapter.provenance()
    if provenance.get("checkpoint_sha256") != specification.checkpoint_sha256:
        raise RuntimeError(f"Checkpoint hash mismatch for {model_id}")
    if provenance.get("container_sha256") != specification.container_digest:
        raise RuntimeError(f"Container hash mismatch for {model_id}")
    prediction_cache.commit()
    model_cache_v1.commit() if model_id not in {"carp_640m", "vespag"} else None
    model_cache_v2.commit() if model_id in {"carp_640m", "vespag"} else None
    temporary = Path("/tmp") / f"confirmation-{panel}-{model_id}-{target_offset:04d}"
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
    manifest = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "freeze_id": "confirmation-prediction-freeze-v1",
        "panel_id": panel,
        "model_id": model_id,
        "family": specification.family,
        "target_offset": target_offset,
        "target_count_requested": len(targets),
        "variant_rows_requested": len(variants),
        "prediction_rows": len(predictions),
        "substitution_coverage": len(predictions) / max(len(variants), 1),
        "cache_hit_count": int(audit["cache_hit"].astype(bool).sum()),
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
        "inputs": input_hashes,
        "artifacts": {
            prediction_path.name: _sha256(prediction_path),
            audit_path.name: _sha256(audit_path),
            provenance_path.name: _sha256(provenance_path),
        },
        "confirmation_outcomes_accessed": False,
    }
    manifest_path = temporary / "execution-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    prediction_volume.reload()
    if destination.exists():
        raise RuntimeError(f"Immutable prediction shard already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(temporary, destination)
    prediction_volume.commit()
    return manifest


@app.function(
    image=fair_image,
    gpu="L4",
    cpu=4,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={
        "/predictions": prediction_volume,
        "/prediction-cache": prediction_cache,
        "/model-cache": model_cache_v1,
    },
)
def score_fair(model_id: str, panel: str, offset: int, limit: int) -> dict[str, object]:
    if model_id not in {
        "esm1v_ensemble",
        "esm2_8m",
        "esm2_35m",
        "esm2_150m",
        "esm2_650m",
        "esm_if1",
    }:
        raise ValueError(f"Unsupported fair-esm model: {model_id}")
    return _execute(model_id, panel, offset, limit)


@app.function(
    image=external_image,
    gpu="L4",
    cpu=8,
    memory=49152,
    timeout=24 * 60 * 60,
    volumes={
        "/predictions": prediction_volume,
        "/prediction-cache": prediction_cache,
        "/model-cache": model_cache_v1,
    },
)
def score_external(model_id: str, panel: str, offset: int, limit: int) -> dict[str, object]:
    if model_id not in {"saprot_35m", "tranception_l_no_retrieval"}:
        raise ValueError(f"Unsupported external model: {model_id}")
    return _execute(model_id, panel, offset, limit)


@app.function(
    image=extension_image,
    gpu="L4",
    cpu=8,
    memory=49152,
    timeout=24 * 60 * 60,
    volumes={
        "/predictions": prediction_volume,
        "/prediction-cache": prediction_cache,
        "/model-cache": model_cache_v2,
    },
)
def score_extension(model_id: str, panel: str, offset: int, limit: int) -> dict[str, object]:
    if model_id not in {"carp_640m", "vespag"}:
        raise ValueError(f"Unsupported extension model: {model_id}")
    return _execute(model_id, panel, offset, limit)


merge_image = modal.Image.debian_slim().pip_install("numpy==1.26.4", "pandas==2.2.3")


@app.function(
    image=merge_image,
    cpu=4,
    memory=16384,
    timeout=60 * 60,
    volumes={"/predictions": prediction_volume},
)
def merge_model(model_id: str, panel: str, expected_targets: int) -> dict[str, object]:
    import shutil

    import pandas as pd

    source = RUN_ROOT / panel / model_id
    shards = sorted(path for path in (source / "shards").iterdir() if path.is_dir())
    if not shards:
        raise RuntimeError(f"No prediction shards exist for {panel}/{model_id}")
    predictions = pd.concat(
        [pd.read_csv(path / "predictions.csv.gz") for path in shards], ignore_index=True
    )
    audits = pd.concat(
        [pd.read_csv(path / "prediction-audit.csv.gz") for path in shards], ignore_index=True
    )
    if audits["target_id"].astype(str).duplicated().any():
        raise RuntimeError("Prediction shards contain duplicate target audits")
    if audits["target_id"].nunique() != expected_targets:
        raise RuntimeError(
            f"Expected {expected_targets} target audits, found {audits['target_id'].nunique()}"
        )
    if predictions.duplicated(["panel_id", "target_id", "variant_id", "model_id"]).any():
        raise RuntimeError("Prediction shards contain duplicate prediction keys")
    manifests = [
        json.loads((path / "execution-manifest.json").read_text(encoding="utf-8"))
        for path in shards
    ]
    input_hashes = {
        json.dumps(item["inputs"], sort_keys=True, separators=(",", ":"))
        for item in manifests
    }
    if len(input_hashes) != 1:
        raise RuntimeError("Prediction shards were produced from inconsistent frozen inputs")
    provenances = [
        json.loads((path / "model-provenance.json").read_text(encoding="utf-8"))[model_id]
        for path in shards
    ]
    if len({json.dumps(item, sort_keys=True) for item in provenances}) != 1:
        raise RuntimeError("Prediction shard provenance is inconsistent")
    temporary = Path("/tmp") / f"confirmation-merge-{panel}-{model_id}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    predictions.to_csv(temporary / "predictions.csv.gz", index=False)
    audits.to_csv(temporary / "prediction-audit.csv.gz", index=False)
    (temporary / "model-provenance.json").write_text(
        json.dumps({model_id: provenances[0]}, indent=2, sort_keys=True) + "\n"
    )
    failures = audits.loc[audits["status"].ne("ok"), ["target_id", "status", "error"]]
    manifest = {
        "schema_version": 1,
        "freeze_id": "confirmation-prediction-freeze-v1",
        "panel_id": panel,
        "model_id": model_id,
        "execution_status": "complete",
        "target_count_requested": expected_targets,
        "variant_rows_requested": int(
            sum(item["variant_rows_requested"] for item in manifests)
        ),
        "prediction_rows": len(predictions),
        "cache_hit_count": int(sum(item["cache_hit_count"] for item in manifests)),
        "elapsed_seconds": float(sum(item["elapsed_seconds"] for item in manifests)),
        "hardware": [item["hardware"] for item in manifests],
        "failures": failures.to_dict(orient="records"),
        "inputs": manifests[0]["inputs"],
        "shard_manifest_hashes": {
            str(path.name): _sha256(path / "execution-manifest.json") for path in shards
        },
        "artifacts": {
            path.name: _sha256(path)
            for path in (
                temporary / "predictions.csv.gz",
                temporary / "prediction-audit.csv.gz",
                temporary / "model-provenance.json",
            )
        },
        "confirmation_outcomes_accessed": False,
    }
    (temporary / "execution-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    for name in (
        "predictions.csv.gz",
        "prediction-audit.csv.gz",
        "model-provenance.json",
        "execution-manifest.json",
    ):
        destination = source / name
        if destination.exists():
            existing = destination
            if _sha256(existing) != _sha256(temporary / name):
                raise RuntimeError(
                    f"Immutable merged prediction differs from retry: {destination}"
                )
            continue
        shutil.copy2(temporary / name, destination)
    prediction_volume.commit()
    return manifest


@app.local_entrypoint()
def main(model_id: str, panel: str, shard_size: int = 10) -> None:
    target_path, *_ = _panel_paths(panel, root=LOCAL_REPOSITORY)
    import pandas as pd

    target_count = len(pd.read_csv(target_path))
    if model_id in {"carp_640m", "vespag"}:
        function = score_extension
    elif model_id in {"saprot_35m", "tranception_l_no_retrieval"}:
        function = score_external
    else:
        function = score_fair
    calls = [
        function.spawn(model_id, panel, offset, min(shard_size, target_count - offset))
        for offset in range(0, target_count, shard_size)
    ]
    for call in calls:
        call.get()
    result = merge_model.remote(model_id, panel, target_count)
    print(json.dumps(result, indent=2, sort_keys=True))
