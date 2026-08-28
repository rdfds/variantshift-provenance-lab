"""Cached wild-type residue embeddings for supervised ESM-2 probes."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .proteingym import read_reference_index
from .provenance import sha256_file

ESM2_MODELS = {
    "esm2_t6_8M_UR50D": ("esm2_t6_8M_UR50D", 6),
    "esm2_t12_35M_UR50D": ("esm2_t12_35M_UR50D", 12),
}


def sequence_digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def embedding_cache_path(output_dir: Path, model_name: str, sequence: str) -> Path:
    return Path(output_dir) / f"{model_name}-{sequence_digest(sequence)[:20]}.npz"


def _window_ranges(length: int, *, window_size: int, overlap: int) -> list[tuple[int, int]]:
    if length < 1:
        raise ValueError("Protein sequence cannot be empty")
    if not 0 <= overlap < window_size:
        raise ValueError("Overlap must lie in [0, window_size)")
    if length <= window_size:
        return [(0, length)]
    step = window_size - overlap
    starts = list(range(0, max(1, length - window_size + 1), step))
    final_start = length - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return [(start, min(length, start + window_size)) for start in starts]


def _load_model(model_name: str, device: str):
    if model_name not in ESM2_MODELS:
        raise ValueError(f"Unsupported ESM-2 embedding model: {model_name}")
    try:
        import esm
    except ImportError as error:
        raise RuntimeError("Install VariantShift with the 'plm' extra") from error
    loader_name, layer = ESM2_MODELS[model_name]
    loader = getattr(esm.pretrained, loader_name)
    model, alphabet = loader()
    model = model.eval().to(device)
    return model, alphabet, layer


def _default_device() -> str:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Install VariantShift with the 'plm' extra") from error
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def embed_sequence(
    sequence: str,
    *,
    model,
    alphabet,
    layer: int,
    device: str,
    window_size: int = 1022,
    overlap: int = 128,
) -> np.ndarray:
    """Embed every residue, averaging representations in overlapping long-sequence windows."""
    import torch

    ranges = _window_ranges(len(sequence), window_size=window_size, overlap=overlap)
    batch_converter = alphabet.get_batch_converter()
    summed: np.ndarray | None = None
    counts = np.zeros(len(sequence), dtype=np.float32)
    with torch.inference_mode():
        for start, end in ranges:
            chunk = sequence[start:end]
            _, _, tokens = batch_converter([("protein", chunk)])
            tokens = tokens.to(device)
            output = model(tokens, repr_layers=[layer], return_contacts=False)
            representation = (
                output["representations"][layer][0, 1 : len(chunk) + 1]
                .float()
                .cpu()
                .numpy()
            )
            if summed is None:
                summed = np.zeros((len(sequence), representation.shape[1]), dtype=np.float32)
            summed[start:end] += representation
            counts[start:end] += 1
    if summed is None or np.any(counts == 0):
        raise RuntimeError("ESM-2 embedding did not cover every residue")
    return summed / counts[:, None]


def build_embedding_cache(
    reference_path: Path,
    eligibility: pd.DataFrame,
    output_dir: Path,
    *,
    model_name: str = "esm2_t6_8M_UR50D",
    device: str | None = None,
) -> pd.DataFrame:
    """Cache one residue-embedding matrix per distinct eligible target sequence."""
    device = device or _default_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = read_reference_index(reference_path)
    eligible = set(
        eligibility.loc[eligibility["eligible"].astype(bool), "assay_id"].astype(str)
    )
    selected = reference.loc[reference["DMS_id"].astype(str).isin(eligible)].copy()
    if selected.empty:
        raise ValueError("No eligible assays were found in the reference index")
    model, alphabet, layer = _load_model(model_name, device)
    paths: dict[str, Path] = {}
    dimensions: dict[str, int] = {}
    for sequence in selected["target_seq"].astype(str).drop_duplicates():
        digest = sequence_digest(sequence)
        path = embedding_cache_path(output_dir, model_name, sequence)
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                embedding = archive["embedding"]
            if embedding.shape[0] != len(sequence):
                raise ValueError(f"Cached embedding length mismatch: {path}")
        else:
            embedding = embed_sequence(
                sequence,
                model=model,
                alphabet=alphabet,
                layer=layer,
                device=device,
            )
            np.savez_compressed(
                path,
                embedding=embedding.astype(np.float32),
                sequence_sha256=np.asarray(digest),
                model_name=np.asarray(model_name),
            )
        paths[digest] = path
        dimensions[digest] = int(embedding.shape[1])

    rows = []
    for metadata in selected.itertuples(index=False):
        sequence = str(metadata.target_seq)
        digest = sequence_digest(sequence)
        rows.append(
            {
                "assay_id": str(metadata.DMS_id),
                "uniprot_id": str(metadata.UniProt_ID),
                "sequence_sha256": digest,
                "sequence_length": len(sequence),
                "model": model_name,
                "embedding_dimension": dimensions[digest],
                "embedding_path": str(paths[digest]),
                "embedding_bytes": paths[digest].stat().st_size,
                "embedding_sha256": sha256_file(paths[digest]),
                "device": device,
            }
        )
    index = pd.DataFrame(rows).sort_values("assay_id").reset_index(drop=True)
    index.to_csv(output_dir / "index.csv", index=False)
    return index


def load_cached_embedding(path: Path, sequence: str) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        embedding = archive["embedding"].astype(np.float32, copy=False)
        digest = str(archive["sequence_sha256"].item())
    if digest != sequence_digest(sequence) or embedding.shape[0] != len(sequence):
        raise ValueError(f"Embedding provenance mismatch: {path}")
    return embedding
