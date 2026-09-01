"""Reproducible model adapters, preflight gates, and prediction caching."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .provenance import environment_versions, sha256_file
from .schemas import (
    PREDICTION_SCHEMA,
    VARIANT_SCHEMA,
    stable_frame_sha256,
    validate_targets,
    write_table,
)


@dataclass(frozen=True)
class ModelSpecification:
    model_id: str
    model_version: str
    family: str
    modalities: tuple[str, ...]
    adapter: str
    source_url: str
    license_name: str
    license_status: str
    checkpoint: str | None = None
    checkpoint_revision: str | None = None
    checkpoint_sha256: str | None = None
    strategy: str | None = None
    command: tuple[str, ...] = ()
    parity_required: bool = False
    training_cutoff: str = "undocumented"
    exposure_status: str = "undocumented"
    container_image: str | None = None
    container_digest: str | None = None
    structure_dir: str | None = None
    input_manifest: str | None = None
    batch_size: int = 16
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelSpecification:
        return cls(
            model_id=str(payload["model_id"]),
            model_version=str(payload["model_version"]),
            family=str(payload["family"]),
            modalities=tuple(map(str, payload["modalities"])),
            adapter=str(payload["adapter"]),
            source_url=str(payload["source_url"]),
            license_name=str(payload["license_name"]),
            license_status=str(payload["license_status"]),
            checkpoint=(str(payload["checkpoint"]) if payload.get("checkpoint") else None),
            checkpoint_revision=(
                str(payload["checkpoint_revision"])
                if payload.get("checkpoint_revision")
                else None
            ),
            checkpoint_sha256=(
                str(payload["checkpoint_sha256"]) if payload.get("checkpoint_sha256") else None
            ),
            strategy=(str(payload["strategy"]) if payload.get("strategy") else None),
            command=tuple(map(str, payload.get("command", []))),
            parity_required=bool(payload.get("parity_required", False)),
            training_cutoff=str(payload.get("training_cutoff", "undocumented")),
            exposure_status=str(payload.get("exposure_status", "undocumented")),
            container_image=(
                str(payload["container_image"]) if payload.get("container_image") else None
            ),
            container_digest=(
                str(payload["container_digest"]) if payload.get("container_digest") else None
            ),
            structure_dir=(str(payload["structure_dir"]) if payload.get("structure_dir") else None),
            input_manifest=(
                str(payload["input_manifest"]) if payload.get("input_manifest") else None
            ),
            batch_size=int(payload.get("batch_size", 16)),
            notes=str(payload.get("notes", "")),
        )

    def digest(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ModelAdapter(ABC):
    """Uniform inference interface; adapters never receive experimental outcomes."""

    def __init__(self, specification: ModelSpecification):
        self.specification = specification

    @abstractmethod
    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        """Return ``variant_id`` and ``score`` for one target."""

    def _resolved_checkpoint_paths(self) -> list[Path]:
        """Return exact checkpoint files loaded by this adapter when locally resolvable."""
        checkpoint = self.specification.checkpoint
        if not checkpoint:
            return []
        direct = Path(checkpoint)
        if direct.is_file():
            return [direct]
        torch_root = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
        names = [item.strip() for item in checkpoint.split(";") if item.strip()]
        candidates = [torch_root / "hub" / "checkpoints" / f"{name}.pt" for name in names]
        return candidates if candidates and all(path.is_file() for path in candidates) else []

    @staticmethod
    def _checkpoint_tree_sha256(paths: list[Path]) -> str | None:
        """Hash checkpoint contents and stable logical names, following snapshot symlinks."""
        files: list[tuple[str, Path]] = []
        for root in paths:
            if root.is_file():
                files.append((root.name, root.resolve()))
            elif root.is_dir():
                files.extend(
                    (path.relative_to(root).as_posix(), path.resolve())
                    for path in sorted(root.rglob("*"))
                    if path.is_file()
                )
        if not files:
            return None
        digest = hashlib.sha256()
        for logical_name, path in files:
            digest.update(logical_name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(sha256_file(path).encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()

    def provenance(self) -> dict[str, object]:
        checkpoint_hash = self.specification.checkpoint_sha256
        resolved_checkpoint_hash = self._checkpoint_tree_sha256(
            self._resolved_checkpoint_paths()
        )
        if resolved_checkpoint_hash:
            checkpoint_hash = resolved_checkpoint_hash
        container_hash = (
            os.environ.get("VARIANTSHIFT_CONTAINER_SHA256")
            or self.specification.container_digest
        )
        if self.specification.container_image:
            container_path = Path(self.specification.container_image)
            if container_path.is_file():
                container_hash = sha256_file(container_path)
        return {
            "specification": asdict(self.specification),
            "specification_sha256": self.specification.digest(),
            "checkpoint_sha256": checkpoint_hash,
            "container_sha256": container_hash,
            "input_manifest_sha256": (
                sha256_file(Path(self.specification.input_manifest))
                if self.specification.input_manifest
                and Path(self.specification.input_manifest).is_file()
                else None
            ),
            "environment": environment_versions(),
        }


class FairESMAdapter(ModelAdapter):
    """Local fair-esm marginal scoring for ESM-1v and ESM-2 checkpoints."""

    def __init__(self, specification: ModelSpecification):
        super().__init__(specification)
        self._runtime: tuple[Any, Any, str] | None = None

    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        from .plm import (
            esm2_position_log_probabilities,
            load_fair_esm_runtime,
            score_single_substitutions,
        )

        loader = self.specification.checkpoint
        strategy = self.specification.strategy or "masked-marginal"
        if not loader:
            raise ValueError("fair-esm adapters require a checkpoint loader name")
        if self._runtime is None:
            self._runtime = load_fair_esm_runtime(loader)
        positions = sorted(variants["position"].astype(int).unique())
        probabilities, token_index = esm2_position_log_probabilities(
            str(target["sequence"]),
            positions,
            model_name=loader,
            strategy=strategy,
            batch_size=max(1, int(self.specification.batch_size)),
            runtime=self._runtime,
        )
        scores = score_single_substitutions(
            probabilities,
            variants["variant_id"].astype(str),
            token_index,
        ).rename(columns={"mutation_codes": "variant_id", "prediction": "score"})
        return scores


class FairESMEnsembleAdapter(ModelAdapter):
    """Average marginal scores from an explicitly ordered fair-esm ensemble."""

    def __init__(self, specification: ModelSpecification):
        super().__init__(specification)
        self._members: dict[str, FairESMAdapter] = {}

    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        checkpoints = [
            item.strip()
            for item in str(self.specification.checkpoint or "").split(";")
            if item.strip()
        ]
        if len(checkpoints) < 2:
            raise ValueError("fair-esm ensemble requires at least two checkpoint loaders")
        members = []
        for checkpoint in checkpoints:
            if checkpoint not in self._members:
                member_specification = ModelSpecification(
                    **{
                        **asdict(self.specification),
                        "adapter": "fair_esm",
                        "checkpoint": checkpoint,
                    }
                )
                self._members[checkpoint] = FairESMAdapter(member_specification)
            member = self._members[checkpoint]
            if member._runtime is not None:
                model, _, device = member._runtime
                model.to(device)
            scores = member.score_target(target, variants)
            members.append(scores.set_index("variant_id")["score"].rename(checkpoint))
            # Retain loaded parameters in host memory between targets while keeping only one
            # 650M checkpoint resident on the GPU. This avoids both repeated disk deserialization
            # and a five-model GPU-memory spike.
            if member._runtime is not None and member._runtime[2] == "cuda":
                import torch

                member._runtime[0].to("cpu")
                torch.cuda.empty_cache()
        combined = pd.concat(members, axis=1, join="inner")
        return combined.mean(axis=1).rename("score").reset_index()


class ESMIF1Adapter(ModelAdapter):
    """ESM-IF1 full-mutant likelihood scoring on frozen structures.

    ProteinGym's released ESM-IF1 baseline teacher-forces each complete mutant sequence and uses
    its mean conditional log-likelihood across non-padding tokens.  Reproducing that definition is
    essential: sitewise alternate-versus-reference log odds are a different estimator.
    """

    def __init__(self, specification: ModelSpecification):
        super().__init__(specification)
        self._runtime: tuple[Any, Any, str] | None = None

    def _load_runtime(self) -> tuple[Any, Any, str]:
        import esm
        import torch

        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        return model.eval().to(device), alphabet, device

    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        import esm.inverse_folding
        import torch
        from esm.inverse_folding.util import CoordBatchConverter
        from torch.nn import functional

        if not self.specification.structure_dir:
            raise ValueError("ESM-IF1 requires a frozen structure directory")
        structure = Path(self.specification.structure_dir) / f"{target['target_id']}.pdb"
        if not structure.is_file():
            raise ValueError(f"Frozen structure is unavailable for {target['target_id']}")
        # fair-esm 2.0.0 still treats this argument as a string and calls
        # ``endswith()`` directly rather than accepting ``os.PathLike``.
        coords, native_sequence = esm.inverse_folding.util.load_coords(str(structure), "A")
        if native_sequence != str(target["sequence"]):
            raise ValueError("ESM-IF1 structure sequence differs from the frozen target")
        if self._runtime is None:
            self._runtime = self._load_runtime()
        model, alphabet, device = self._runtime
        converter = CoordBatchConverter(alphabet)
        base = str(target["sequence"])
        rows = variants.loc[:, ["variant_id", "position", "reference", "alternate"]].copy()
        strategy = self.specification.strategy or "full-sequence-average-log-likelihood"
        if strategy != "full-sequence-average-log-likelihood":
            raise ValueError(f"Unsupported ESM-IF1 scoring strategy: {strategy}")
        sequences: list[str] = []
        for row in rows.itertuples(index=False):
            position = int(row.position) - 1
            if base[position] != str(row.reference):
                raise ValueError(
                    f"Variant {row.variant_id} reference differs from the frozen target"
                )
            sequences.append(base[:position] + str(row.alternate) + base[position + 1 :])
        scores: list[float] = []
        # Full-mutant IF1 memory grows steeply with both sequence length and batch size.
        # Cap total residues per batch so long external targets do not turn a validated
        # checkpoint into an avoidable out-of-memory exclusion.
        batch_size = min(
            max(1, int(self.specification.batch_size)),
            max(1, 8192 // len(base)),
        )
        with torch.inference_mode():
            for start in range(0, len(sequences), batch_size):
                current = sequences[start : start + batch_size]
                batch = [(coords, None, sequence) for sequence in current]
                coordinates, confidence, _, tokens, padding_mask = converter(
                    batch, device=device
                )
                target_tokens = tokens[:, 1:]
                logits, _ = model.forward(
                    coordinates,
                    padding_mask,
                    confidence,
                    tokens[:, :-1],
                )
                losses = functional.cross_entropy(logits, target_tokens, reduction="none")
                scored = target_tokens.ne(alphabet.padding_idx)
                likelihoods = -(losses * scored).sum(dim=1) / scored.sum(dim=1)
                scores.extend(likelihoods.float().cpu().tolist())
                del coordinates, confidence, tokens, padding_mask
                del target_tokens, logits, losses, scored, likelihoods
        return pd.DataFrame({"variant_id": rows["variant_id"], "score": scores})


def _external_repository(environment_variable: str, default: str) -> Path:
    root = Path(os.environ.get(environment_variable, default)).expanduser().resolve()
    if not root.is_dir():
        raise RuntimeError(f"External model repository is unavailable: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


class ProteinMPNNAdapter(ModelAdapter):
    """ProteinMPNN full-mutant likelihoods using the ProteinGym scoring path."""

    def __init__(self, specification: ModelSpecification):
        super().__init__(specification)
        self._runtime: tuple[Any, Any, Any, str] | None = None
        self._checkpoint_path: Path | None = None

    def _resolved_checkpoint_paths(self) -> list[Path]:
        return [self._checkpoint_path] if self._checkpoint_path is not None else []

    def _load_runtime(self) -> tuple[Any, Any, Any, str]:
        import torch

        root = _external_repository("PROTEINMPNN_DIR", "third_party/ProteinMPNN")
        import protein_mpnn_utils as mpnn_utils

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        checkpoint_name = self.specification.checkpoint or "v_48_020"
        checkpoint_path = root / "vanilla_model_weights" / f"{checkpoint_name}.pt"
        if not checkpoint_path.is_file():
            raise RuntimeError(f"ProteinMPNN checkpoint is unavailable: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self._checkpoint_path = checkpoint_path
        model = mpnn_utils.ProteinMPNN(
            ca_only=False,
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            augment_eps=0.0,
            k_neighbors=checkpoint["num_edges"],
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model.eval().to(device), mpnn_utils, torch, device

    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        import copy

        if not self.specification.structure_dir:
            raise ValueError("ProteinMPNN requires a frozen structure directory")
        structure = Path(self.specification.structure_dir) / f"{target['target_id']}.pdb"
        if not structure.is_file():
            raise ValueError(f"Frozen structure is unavailable for {target['target_id']}")
        strategy = self.specification.strategy or "full-sequence-autoregressive-log-likelihood"
        if strategy != "full-sequence-autoregressive-log-likelihood":
            raise ValueError(f"Unsupported ProteinMPNN scoring strategy: {strategy}")
        if self._runtime is None:
            self._runtime = self._load_runtime()
        model, mpnn_utils, torch, device = self._runtime
        parsed = mpnn_utils.parse_PDB(str(structure), input_chain_list=["A"], ca_only=False)
        if len(parsed) != 1:
            raise ValueError(f"ProteinMPNN expected one parsed structure for {target['target_id']}")
        record = parsed[0]
        native_sequence = str(record.get("seq_chain_A", record.get("seq", "")))
        if native_sequence != str(target["sequence"]):
            raise ValueError("ProteinMPNN structure sequence differs from the frozen target")
        alphabet = "ACDEFGHIKLMNPQRSTVWYX"
        token_index = {amino_acid: index for index, amino_acid in enumerate(alphabet)}
        base = str(target["sequence"])
        rows = variants.loc[:, ["variant_id", "position", "reference", "alternate"]].copy()
        sequences: list[str] = []
        for row in rows.itertuples(index=False):
            position = int(row.position) - 1
            if base[position] != str(row.reference):
                raise ValueError(
                    f"Variant {row.variant_id} reference differs from the frozen target"
                )
            sequences.append(base[:position] + str(row.alternate) + base[position + 1 :])

        # ProteinGym's published runner draws one autoregressive decoding order per sequence.
        # Deriving the generator seed from the target makes the estimator invariant to sharding
        # while retaining that same one-order scoring definition.
        target_seed = int.from_bytes(
            hashlib.sha256(str(target["target_id"]).encode("utf-8")).digest()[:8],
            "big",
        ) % (2**31)
        generator = torch.Generator(device=device)
        generator.manual_seed(target_seed)
        scores: list[float] = []
        batch_size = max(1, int(self.specification.batch_size))
        chain_dictionary = {record["name"]: (["A"], [])}
        with torch.inference_mode():
            for start in range(0, len(sequences), batch_size):
                current = sequences[start : start + batch_size]
                batch = [copy.deepcopy(record) for _ in current]
                features = mpnn_utils.tied_featurize(
                    batch,
                    device,
                    chain_dictionary,
                    None,
                    None,
                    None,
                    None,
                    None,
                    ca_only=False,
                )
                coordinates = features[0]
                sequence_tokens = features[1]
                mask = features[2]
                chain_mask = features[4]
                chain_encoding = features[5]
                chain_position_mask = features[10]
                residue_index = features[12]
                for index, sequence in enumerate(current):
                    sequence_tokens[index, : len(sequence)] = torch.tensor(
                        [token_index[amino_acid] for amino_acid in sequence],
                        device=device,
                    )
                decoding_noise = torch.randn(
                    chain_mask.shape, device=device, generator=generator
                )
                log_probabilities = model(
                    coordinates,
                    sequence_tokens,
                    mask,
                    chain_mask * chain_position_mask,
                    residue_index,
                    chain_encoding,
                    decoding_noise,
                )
                global_scores = mpnn_utils._scores(
                    sequence_tokens,
                    log_probabilities,
                    mask,
                )
                scores.extend((-global_scores).float().cpu().tolist())
        return pd.DataFrame({"variant_id": rows["variant_id"], "score": scores})


class SaProtAdapter(ModelAdapter):
    """SaProt masked-marginal scores from frozen AlphaFold crops and Foldseek 3Di tokens."""

    _STRUCTURE_VOCAB = "pynwrqhgdlvtmfsaeikc#"

    def __init__(self, specification: ModelSpecification):
        super().__init__(specification)
        self._runtime: tuple[Any, Any, Any, str] | None = None
        self._checkpoint_root: Path | None = None

    def _resolved_checkpoint_paths(self) -> list[Path]:
        return [self._checkpoint_root] if self._checkpoint_root is not None else []

    def _load_runtime(self) -> tuple[Any, Any, Any, str]:
        import torch
        from transformers import EsmForMaskedLM, EsmTokenizer

        checkpoint = self.specification.checkpoint or "westlake-repl/SaProt_35M_AF2"
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        revision = self.specification.checkpoint_revision
        tokenizer = EsmTokenizer.from_pretrained(checkpoint, revision=revision)
        model = EsmForMaskedLM.from_pretrained(checkpoint, revision=revision).eval().to(device)
        snapshot = getattr(model.config, "_name_or_path", "")
        if Path(str(snapshot)).is_dir():
            self._checkpoint_root = Path(str(snapshot))
        else:
            from huggingface_hub import snapshot_download

            self._checkpoint_root = Path(snapshot_download(checkpoint, revision=revision))
        return model, tokenizer, torch, device

    @staticmethod
    def _plddt_by_residue(structure: Path) -> np.ndarray:
        values: dict[tuple[str, str], list[float]] = {}
        for line in structure.read_text(encoding="ascii").splitlines():
            if not line.startswith(("ATOM  ", "HETATM")) or line[21:22] != "A":
                continue
            key = (line[22:26].strip(), line[26:27].strip())
            values.setdefault(key, []).append(float(line[60:66]))
        return np.asarray([np.mean(current) for current in values.values()], dtype=float)

    @staticmethod
    def _foldseek_sequence(structure: Path) -> tuple[str, str]:
        executable = os.environ.get("FOLDSEEK_BINARY") or shutil.which("foldseek")
        if not executable:
            raise RuntimeError("SaProt requires a Foldseek executable")
        with tempfile.TemporaryDirectory(prefix="variantshift-foldseek-") as temporary:
            output = Path(temporary) / "descriptor.tsv"
            subprocess.run(
                [
                    str(executable),
                    "structureto3didescriptor",
                    "-v",
                    "0",
                    "--threads",
                    "1",
                    "--chain-name-mode",
                    "1",
                    str(structure),
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            lines = output.read_text(encoding="utf-8").splitlines()
        rows = [line.split("\t") for line in lines if len(line.split("\t")) >= 3]
        if not rows:
            raise RuntimeError(f"Foldseek produced no 3Di sequence for {structure.name}")
        sequence, structure_sequence = rows[0][1:3]
        return sequence.upper(), structure_sequence.lower()

    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        if not self.specification.structure_dir:
            raise ValueError("SaProt requires a frozen structure directory")
        structure = Path(self.specification.structure_dir) / f"{target['target_id']}.pdb"
        if not structure.is_file():
            raise ValueError(f"Frozen structure is unavailable for {target['target_id']}")
        strategy = self.specification.strategy or "masked-marginal-plddt70"
        if strategy != "masked-marginal-plddt70":
            raise ValueError(f"Unsupported SaProt scoring strategy: {strategy}")
        sequence, structure_sequence = self._foldseek_sequence(structure)
        if sequence != str(target["sequence"]):
            raise ValueError("SaProt structure sequence differs from the frozen target")
        confidence = self._plddt_by_residue(structure)
        if len(confidence) != len(sequence) or len(structure_sequence) != len(sequence):
            raise ValueError("SaProt structure annotations do not cover the frozen sequence")
        structure_tokens = np.asarray(list(structure_sequence), dtype="U1")
        structure_tokens[confidence < 70.0] = "#"
        combined = [amino_acid + token for amino_acid, token in zip(sequence, structure_tokens)]
        if self._runtime is None:
            self._runtime = self._load_runtime()
        model, tokenizer, torch, device = self._runtime
        vocabulary = tokenizer.get_vocab()
        amino_acid_ids = {
            amino_acid: [
                vocabulary[amino_acid + structural] for structural in self._STRUCTURE_VOCAB
            ]
            for amino_acid in "ACDEFGHIKLMNPQRSTVWY"
        }
        requested_positions = sorted(variants["position"].astype(int).unique())
        position_scores: dict[int, dict[str, float]] = {}
        batch_size = max(1, int(self.specification.batch_size))
        with torch.inference_mode():
            for offset in range(0, len(requested_positions), batch_size):
                current = requested_positions[offset : offset + batch_size]
                masked_sequences = []
                for position in current:
                    tokens = combined.copy()
                    tokens[position - 1] = "#" + structure_tokens[position - 1]
                    masked_sequences.append(" ".join(tokens))
                encoded = tokenizer(masked_sequences, return_tensors="pt", padding=True)
                encoded = {key: value.to(device) for key, value in encoded.items()}
                logits = model(**encoded).logits
                for batch_index, position in enumerate(current):
                    residue_logits = logits[batch_index, position]
                    position_scores[position] = {
                        amino_acid: float(
                            torch.logsumexp(
                                residue_logits[torch.tensor(indices, device=device)], dim=0
                            ).cpu()
                        )
                        for amino_acid, indices in amino_acid_ids.items()
                    }
        rows = variants.loc[:, ["variant_id", "position", "reference", "alternate"]]
        scores = [
            position_scores[int(row.position)][str(row.alternate)]
            - position_scores[int(row.position)][str(row.reference)]
            for row in rows.itertuples(index=False)
        ]
        return pd.DataFrame({"variant_id": rows["variant_id"], "score": scores})


class TranceptionAdapter(ModelAdapter):
    """Official Tranception mirrored full-sequence likelihood scoring without retrieval."""

    def __init__(self, specification: ModelSpecification):
        super().__init__(specification)
        self._runtime: tuple[Any, Any] | None = None
        self._checkpoint_root: Path | None = None

    def _resolved_checkpoint_paths(self) -> list[Path]:
        return [self._checkpoint_root] if self._checkpoint_root is not None else []

    def _load_runtime(self) -> tuple[Any, Any]:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import PreTrainedTokenizerFast

        root = _external_repository("TRANCEPTION_DIR", "third_party/Tranception")
        import tranception
        from tranception import model_pytorch

        checkpoint = os.environ.get("TRANCEPTION_CHECKPOINT")
        if not checkpoint:
            checkpoint_name = self.specification.checkpoint or "OATML-Markslab/Tranception_Large"
            checkpoint = snapshot_download(
                checkpoint_name,
                revision=self.specification.checkpoint_revision,
            )
        checkpoint_path = Path(checkpoint)
        self._checkpoint_root = checkpoint_path
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(root / "tranception" / "utils" / "tokenizers" / "Basic_tokenizer"),
            unk_token="[UNK]",
            sep_token="[SEP]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            mask_token="[MASK]",
        )
        configuration = tranception.config.TranceptionConfig(
            **json.loads((checkpoint_path / "config.json").read_text(encoding="utf-8"))
        )
        configuration.attention_mode = "tranception"
        configuration.position_embedding = "grouped_alibi"
        configuration.tokenizer = tokenizer
        configuration.scoring_window = "optimal"
        configuration.retrieval_aggregation_mode = None
        model = model_pytorch.TranceptionLMHeadModel.from_pretrained(
            pretrained_model_name_or_path=str(checkpoint_path),
            config=configuration,
        )
        if torch.cuda.is_available():
            model = model.cuda()
        return model.eval(), torch

    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        strategy = self.specification.strategy or "mirrored-no-retrieval"
        if strategy != "mirrored-no-retrieval":
            raise ValueError(f"Unsupported Tranception scoring strategy: {strategy}")
        if self._runtime is None:
            self._runtime = self._load_runtime()
        model, _ = self._runtime
        base = str(target["sequence"])
        rows = variants.loc[:, ["variant_id", "position", "alternate"]].copy()
        rows["mutant"] = rows["variant_id"]
        rows["mutated_sequence"] = [
            f"{base[: int(row.position) - 1]}{row.alternate}{base[int(row.position) :]}"
            for row in rows.itertuples(index=False)
        ]
        # The large checkpoint's activation memory scales with both sequence length and batch
        # size. Keep roughly 4,096 residues per inference batch so the exact same scorer covers
        # long parity proteins and short Domainome targets without target-specific manual tuning.
        inference_batch_size = min(
            max(1, int(self.specification.batch_size)),
            max(1, 4096 // len(base)),
        )
        scored = model.score_mutants(
            DMS_data=rows.loc[:, ["mutant", "mutated_sequence"]],
            target_seq=base,
            scoring_mirror=True,
            batch_size_inference=inference_batch_size,
            num_workers=0,
            indel_mode=False,
        )
        aligned = rows.merge(
            scored.loc[:, ["mutated_sequence", "avg_score"]],
            on="mutated_sequence",
            how="left",
            validate="one_to_one",
        )
        return aligned.loc[:, ["variant_id", "avg_score"]].rename(
            columns={"avg_score": "score"}
        )


class ProSSTAdapter(ModelAdapter):
    """Official ProSST-2048 zero-shot log-odds using frozen PDB structures.

    The structure tokenizer is part of the pinned ProSST repository.  Its autoencoder and
    2,048-state clustering model are treated as checkpoint material and therefore included in
    the adapter's resolved checkpoint hash together with the Hugging Face snapshot.
    """

    def __init__(self, specification: ModelSpecification):
        super().__init__(specification)
        self._runtime: tuple[Any, Any, Any, str] | None = None
        self._checkpoint_paths: list[Path] = []
        self._structure_predictor: Any | None = None

    def _resolved_checkpoint_paths(self) -> list[Path]:
        return self._checkpoint_paths

    def _load_runtime(self) -> tuple[Any, Any, Any, str]:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        root = _external_repository("PROSST_DIR", "third_party/ProSST")
        checkpoint = self.specification.checkpoint or "AI4Protein/ProSST-2048"
        revision = self.specification.checkpoint_revision
        snapshot = Path(snapshot_download(checkpoint, revision=revision))
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot, trust_remote_code=True, local_files_only=True
        )
        model = AutoModelForMaskedLM.from_pretrained(
            snapshot, trust_remote_code=True, local_files_only=True
        )
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        model = model.eval().to(device)
        static = root / "prosst" / "structure" / "static"
        autoencoder = static / "AE.pt"
        cluster = static / "2048.joblib"
        if not autoencoder.is_file() or not cluster.is_file():
            raise RuntimeError("Pinned ProSST structure-quantizer weights are unavailable")
        self._checkpoint_paths = [snapshot, autoencoder, cluster]
        return model, tokenizer, torch, device

    def _structure_tokens(self, structure: Path, sequence: str) -> list[int]:
        if self._structure_predictor is None:
            import torch

            _external_repository("PROSST_DIR", "third_party/ProSST")
            from prosst.structure.get_sst_seq import SSTPredictor

            self._structure_predictor = SSTPredictor(
                structure_vocab_size=2048,
                num_processes=1,
                num_threads=1,
                device=("cuda" if torch.cuda.is_available() else "cpu"),
            )
        records = self._structure_predictor.predict_from_pdb([str(structure)])
        if len(records) != 1:
            raise RuntimeError(f"ProSST returned {len(records)} structure records")
        record = records[0]
        observed = str(record["aa_seq"]).upper()
        if observed != sequence:
            raise ValueError("ProSST structure sequence differs from the frozen target")
        token_key = next(
            (key for key in record if key.endswith("2048_sst_seq")), None
        )
        if token_key is None:
            raise RuntimeError("ProSST did not return the frozen 2,048-state tokenization")
        tokens = [int(value) for value in record[token_key]]
        if len(tokens) != len(sequence):
            raise ValueError("ProSST structure tokens do not cover the frozen sequence")
        return tokens

    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        if not self.specification.structure_dir:
            raise ValueError("ProSST requires a frozen structure directory")
        structure = Path(self.specification.structure_dir) / f"{target['target_id']}.pdb"
        if not structure.is_file():
            raise ValueError(f"Frozen structure is unavailable for {target['target_id']}")
        strategy = self.specification.strategy or "wild-type-marginal-structure-2048"
        if strategy != "wild-type-marginal-structure-2048":
            raise ValueError(f"Unsupported ProSST scoring strategy: {strategy}")
        if self._runtime is None:
            self._runtime = self._load_runtime()
        model, tokenizer, torch, device = self._runtime
        sequence = str(target["sequence"])
        structure_tokens = self._structure_tokens(structure, sequence)
        encoded = tokenizer([sequence], return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        ss_input_ids = torch.tensor(
            [[1, *[value + 3 for value in structure_tokens], 2]],
            dtype=torch.long,
            device=device,
        )
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ss_input_ids=ss_input_ids,
            ).logits
            log_probabilities = torch.log_softmax(logits[:, 1:-1, :], dim=-1)[0]
        vocabulary = tokenizer.get_vocab()
        rows = variants.loc[:, ["variant_id", "position", "reference", "alternate"]]
        scores = []
        for row in rows.itertuples(index=False):
            position = int(row.position) - 1
            if sequence[position] != str(row.reference):
                raise ValueError(
                    f"Variant {row.variant_id} reference differs from the frozen target"
                )
            scores.append(
                float(
                    log_probabilities[position, vocabulary[str(row.alternate)]]
                    - log_probabilities[position, vocabulary[str(row.reference)]]
                )
            )
        return pd.DataFrame({"variant_id": rows["variant_id"], "score": scores})


class VespaGAdapter(ModelAdapter):
    """VespaG v2 raw GEMME-distillation scores from official ESM-2 3B embeddings."""

    _ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

    def __init__(self, specification: ModelSpecification):
        super().__init__(specification)
        self._runtime: tuple[Any, Any, Any, Any, str] | None = None
        self._checkpoint_paths: list[Path] = []

    def _resolved_checkpoint_paths(self) -> list[Path]:
        return self._checkpoint_paths

    def _load_runtime(self) -> tuple[Any, Any, Any, Any, str]:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        root = _external_repository("VESPAG_DIR", "third_party/VespaG")
        from vespag.models import FNN

        encoder_name = self.specification.checkpoint or "facebook/esm2_t36_3B_UR50D"
        encoder_revision = self.specification.checkpoint_revision
        snapshot = Path(snapshot_download(encoder_name, revision=encoder_revision))
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        encoder = AutoModel.from_pretrained(snapshot, local_files_only=True)
        predictor_checkpoint = root / "model_weights" / "v2" / "esm2.pt"
        if not predictor_checkpoint.is_file():
            raise RuntimeError("Pinned VespaG v2 ESM-2 predictor checkpoint is unavailable")
        predictor = FNN(
            hidden_layer_sizes=[256],
            input_dim=2560,
            output_dim=20,
            dropout_rate=0.2,
        )
        predictor.load_state_dict(torch.load(predictor_checkpoint, map_location="cpu"))
        if torch.cuda.is_available():
            device = "cuda"
            encoder = encoder.half()
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        encoder = encoder.eval().to(device)
        predictor = predictor.eval().float().to(device)
        self._checkpoint_paths = [snapshot, predictor_checkpoint]
        return encoder, tokenizer, predictor, torch, device

    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        strategy = self.specification.strategy or "raw-esm2-3b-fnn-v2"
        if strategy != "raw-esm2-3b-fnn-v2":
            raise ValueError(f"Unsupported VespaG scoring strategy: {strategy}")
        if self._runtime is None:
            self._runtime = self._load_runtime()
        encoder, tokenizer, predictor, torch, device = self._runtime
        sequence = str(target["sequence"])
        if len(sequence) > 1022:
            raise ValueError(
                "VespaG's official embedder has no validated overlapping-window protocol for "
                "sequences longer than 1,022 residues"
            )
        input_sequence = " ".join(list(sequence.translate(str.maketrans("UZOB", "XXXX"))))
        tokens = tokenizer(
            [input_sequence], add_special_tokens=True, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            embeddings = encoder(**tokens).last_hidden_state[:, 1 : len(sequence) + 1]
            scores_by_position = predictor(embeddings.float()).squeeze(0).cpu().numpy()
        alphabet_index = {amino_acid: index for index, amino_acid in enumerate(self._ALPHABET)}
        rows = variants.loc[:, ["variant_id", "position", "reference", "alternate"]]
        scores = []
        for row in rows.itertuples(index=False):
            position = int(row.position) - 1
            if sequence[position] != str(row.reference):
                raise ValueError(
                    f"Variant {row.variant_id} reference differs from the frozen target"
                )
            scores.append(float(scores_by_position[position, alphabet_index[str(row.alternate)]]))
        return pd.DataFrame({"variant_id": rows["variant_id"], "score": scores})


class CARPAdapter(ModelAdapter):
    """ProteinGym-compatible CARP-640M convolutional masked-marginal scores."""

    def __init__(self, specification: ModelSpecification):
        super().__init__(specification)
        self._runtime: tuple[Any, Any, Any, str, str] | None = None
        self._checkpoint_path: Path | None = None

    def _resolved_checkpoint_paths(self) -> list[Path]:
        return [self._checkpoint_path] if self._checkpoint_path is not None else []

    def _load_runtime(self) -> tuple[Any, Any, Any, str, str]:
        import torch

        _external_repository("CARP_DIR", "third_party/protein-sequence-models")
        from sequence_models.constants import MASK
        from sequence_models.pretrained import load_model_and_alphabet

        checkpoint_name = self.specification.checkpoint or "carp_640M"
        checkpoint_dir = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch"))
        checkpoint_dir = checkpoint_dir / "hub" / "checkpoints"
        model, collater = load_model_and_alphabet(checkpoint_name)
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}.pt"
        if not checkpoint_path.is_file():
            raise RuntimeError(f"CARP checkpoint is unavailable after loading: {checkpoint_path}")
        self._checkpoint_path = checkpoint_path
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        return model.eval().to(device), collater, torch, device, MASK

    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        strategy = self.specification.strategy or "masked-marginal"
        if strategy != "masked-marginal":
            raise ValueError(f"Unsupported CARP scoring strategy: {strategy}")
        if self._runtime is None:
            self._runtime = self._load_runtime()
        model, collater, torch, device, mask_token = self._runtime
        from sequence_models.constants import PROTEIN_ALPHABET

        sequence = str(target["sequence"])
        input_ids = collater([[sequence]])[0]
        mask_index = PROTEIN_ALPHABET.index(mask_token)
        requested_positions = sorted(variants["position"].astype(int).unique())
        position_scores: dict[int, Any] = {}
        batch_size = min(
            max(1, int(self.specification.batch_size)),
            max(1, 8192 // len(sequence)),
        )
        with torch.inference_mode():
            for start in range(0, len(requested_positions), batch_size):
                positions = requested_positions[start : start + batch_size]
                current = input_ids.repeat(len(positions), 1)
                for batch_index, position in enumerate(positions):
                    current[batch_index, position - 1] = mask_index
                logits = model(current.to(device), logits=True)["logits"]
                probabilities = torch.log_softmax(logits, dim=-1).cpu()
                for batch_index, position in enumerate(positions):
                    position_scores[position] = probabilities[batch_index, position - 1]
        rows = variants.loc[:, ["variant_id", "position", "reference", "alternate"]]
        scores = []
        for row in rows.itertuples(index=False):
            position = int(row.position)
            if sequence[position - 1] != str(row.reference):
                raise ValueError(
                    f"Variant {row.variant_id} reference differs from the frozen target"
                )
            current = position_scores[position]
            scores.append(
                float(
                    current[PROTEIN_ALPHABET.index(str(row.alternate))]
                    - current[PROTEIN_ALPHABET.index(str(row.reference))]
                )
            )
        return pd.DataFrame({"variant_id": rows["variant_id"], "score": scores})


class CommandModelAdapter(ModelAdapter):
    """Run a pinned external scorer through a narrow CSV contract without a shell."""

    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        if not self.specification.command:
            raise ValueError("Command adapter requires a non-empty command array")
        executable = self.specification.command[0]
        if shutil.which(executable) is None and not Path(executable).is_file():
            raise RuntimeError(f"Model executable is unavailable: {executable}")
        with tempfile.TemporaryDirectory(prefix="variantshift-model-") as temporary:
            root = Path(temporary)
            input_path = root / "input.csv"
            output_path = root / "output.csv"
            input_frame = variants.copy()
            input_frame["sequence"] = str(target["sequence"])
            input_frame.to_csv(input_path, index=False)
            replacements = {
                "{input}": str(input_path),
                "{output}": str(output_path),
                "{target_id}": str(target["target_id"]),
            }
            command = [replacements.get(token, token) for token in self.specification.command]
            subprocess.run(command, check=True, capture_output=True, text=True)
            if not output_path.is_file():
                raise RuntimeError("Model command did not create its declared output CSV")
            scores = pd.read_csv(output_path)
        missing = {"variant_id", "score"}.difference(scores.columns)
        if missing:
            raise ValueError(f"Model output is missing columns: {sorted(missing)}")
        return scores.loc[:, ["variant_id", "score"]]


class PrecomputedModelAdapter(ModelAdapter):
    def __init__(self, specification: ModelSpecification, predictions: Path):
        super().__init__(specification)
        self.predictions = pd.read_csv(predictions)

    def score_target(self, target: pd.Series, variants: pd.DataFrame) -> pd.DataFrame:
        rows = self.predictions.loc[
            self.predictions["target_id"].astype(str).eq(str(target["target_id"]))
        ]
        if rows.empty:
            raise ValueError(f"No precomputed predictions for target {target['target_id']}")
        return rows.loc[:, ["variant_id", "score"]].copy()


def load_model_specifications(path: Path) -> list[ModelSpecification]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    specifications = [ModelSpecification.from_dict(item) for item in payload["models"]]
    identifiers = [item.model_id for item in specifications]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Model identifiers must be unique")
    return specifications


def adapter_from_specification(
    specification: ModelSpecification,
    *,
    precomputed_path: Path | None = None,
) -> ModelAdapter:
    if specification.adapter == "fair_esm":
        return FairESMAdapter(specification)
    if specification.adapter == "fair_esm_ensemble":
        return FairESMEnsembleAdapter(specification)
    if specification.adapter == "esm_if1":
        return ESMIF1Adapter(specification)
    if specification.adapter == "protein_mpnn":
        return ProteinMPNNAdapter(specification)
    if specification.adapter == "saprot":
        return SaProtAdapter(specification)
    if specification.adapter == "tranception":
        return TranceptionAdapter(specification)
    if specification.adapter == "prosst":
        return ProSSTAdapter(specification)
    if specification.adapter == "vespag":
        return VespaGAdapter(specification)
    if specification.adapter == "carp":
        return CARPAdapter(specification)
    if specification.adapter == "command":
        return CommandModelAdapter(specification)
    if specification.adapter == "precomputed":
        if precomputed_path is None:
            raise ValueError("Precomputed adapter requires a prediction path")
        return PrecomputedModelAdapter(specification, precomputed_path)
    raise ValueError(f"Unsupported model adapter: {specification.adapter}")


def prediction_cache_key(
    specification: ModelSpecification,
    target: pd.Series,
    variants: pd.DataFrame,
) -> str:
    payload = {
        "model": specification.digest(),
        "target": str(target["sequence_sha256"]),
        "variants": stable_frame_sha256(variants.loc[:, VARIANT_SCHEMA.required]),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def score_panel(
    adapter: ModelAdapter,
    targets: pd.DataFrame,
    variants: pd.DataFrame,
    *,
    protocol_id: str,
    cache_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a target panel with content-addressed, resumable target jobs."""
    validate_targets(targets)
    VARIANT_SCHEMA.validate(variants)
    cache_dir = Path(cache_dir) / adapter.specification.model_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for _, target in targets.sort_values("target_id").iterrows():
        target_variants = variants.loc[
            variants["target_id"].astype(str).eq(str(target["target_id"]))
        ].copy()
        key = prediction_cache_key(adapter.specification, target, target_variants)
        cached = cache_dir / f"{key}.csv"
        cache_hit = cached.is_file()
        started = time.perf_counter()
        try:
            if cache_hit:
                scores = pd.read_csv(cached)
            else:
                scores = adapter.score_target(target, target_variants)
                scores.to_csv(cached, index=False, lineterminator="\n")
            aligned = target_variants.loc[:, ["panel_id", "target_id", "variant_id"]].merge(
                scores.loc[:, ["variant_id", "score"]],
                on="variant_id",
                how="left",
                validate="one_to_one",
            )
            aligned["protocol_id"] = protocol_id
            aligned["model_id"] = adapter.specification.model_id
            aligned["model_version"] = adapter.specification.model_version
            aligned["status"] = np.where(aligned["score"].notna(), "ok", "missing")
            prediction_rows.append(aligned.loc[:, PREDICTION_SCHEMA.required])
            coverage = float(aligned["score"].notna().mean())
            status = "ok"
            error = ""
        except Exception as exception:  # noqa: BLE001 - isolate one external model failure
            coverage = 0.0
            status = "failed"
            error = f"{type(exception).__name__}: {exception}"
        elapsed_seconds = time.perf_counter() - started
        audit_rows.append(
            {
                "model_id": adapter.specification.model_id,
                "target_id": target["target_id"],
                "sequence_sha256": target["sequence_sha256"],
                "cache_key": key,
                "cache_hit": cache_hit,
                "coverage": coverage,
                "elapsed_seconds": elapsed_seconds,
                "variants_per_second": (
                    len(target_variants) / elapsed_seconds if elapsed_seconds > 0 else np.nan
                ),
                "status": status,
                "error": error,
            }
        )
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    if not prediction_rows:
        failures = "; ".join(
            f"{row['target_id']}: {row['error']}" for row in audit_rows[:3]
        )
        raise RuntimeError(
            f"Model {adapter.specification.model_id} failed on every target; "
            f"first failures: {failures}"
        )
    predictions = pd.concat(prediction_rows, ignore_index=True)
    PREDICTION_SCHEMA.validate(predictions)
    return predictions, pd.DataFrame(audit_rows)


def _parity_correlation(predictions: pd.DataFrame, reference: pd.DataFrame) -> float:
    shared = predictions.merge(
        reference.loc[:, ["target_id", "variant_id", "score"]].rename(
            columns={"score": "reference_score"}
        ),
        on=["target_id", "variant_id"],
        how="inner",
    )
    if len(shared) < 2:
        return float("nan")
    return float(spearmanr(shared["score"], shared["reference_score"]).statistic)


def preflight_models(
    specifications: list[ModelSpecification],
    *,
    targets: pd.DataFrame | None = None,
    variants: pd.DataFrame | None = None,
    protocol_id: str = "preflight",
    cache_dir: Path = Path("artifacts/model-cache"),
    parity_dir: Path | None = None,
    execute: bool = False,
) -> pd.DataFrame:
    """Apply deterministic metadata, coverage, parity, and repeatability gates."""
    rows: list[dict[str, object]] = []
    passing_targets_by_model: dict[str, set[str]] = {}
    for specification in specifications:
        row: dict[str, object] = {
            "model_id": specification.model_id,
            "model_version": specification.model_version,
            "family": specification.family,
            "modalities": ";".join(specification.modalities),
            "adapter": specification.adapter,
            "source_url": specification.source_url,
            "license_name": specification.license_name,
            "license_status": specification.license_status,
            "training_cutoff": specification.training_cutoff,
            "exposure_status": specification.exposure_status,
            "container_image": specification.container_image,
            "container_digest": specification.container_digest,
            "checkpoint_sha256": specification.checkpoint_sha256,
            "provenance_complete": False,
            "metadata_complete": bool(
                specification.source_url
                and specification.license_name
                and specification.license_status in {"permitted", "restricted", "prohibited"}
            ),
            "execution_status": "not_run",
            "coverage": np.nan,
            "parity_spearman": np.nan,
            "repeat_spearman": np.nan,
            "elapsed_seconds": np.nan,
            "variants_per_second": np.nan,
            "target_count": 0 if targets is None else int(targets["target_id"].nunique()),
            "targets_at_95pct_coverage": 0,
            "primary_eligible": False,
            "exclusion_reason": "execution_not_requested",
        }
        if not execute:
            rows.append(row)
            continue
        if targets is None or variants is None:
            raise ValueError("Executable preflight requires target and variant tables")
        try:
            adapter = adapter_from_specification(specification)
            first, first_audit = score_panel(
                adapter,
                targets,
                variants,
                protocol_id=protocol_id,
                cache_dir=cache_dir / "first",
            )
            second, _ = score_panel(
                adapter,
                targets,
                variants,
                protocol_id=protocol_id,
                cache_dir=cache_dir / "second",
            )
            merged = first.merge(
                second,
                on=["target_id", "variant_id", "model_id"],
                suffixes=("_first", "_second"),
            )
            row["coverage"] = float(first["score"].notna().mean())
            passing_targets = set(
                first_audit.loc[
                    first_audit["status"].eq("ok") & first_audit["coverage"].ge(0.95),
                    "target_id",
                ].astype(str)
            )
            passing_targets_by_model[specification.model_id] = passing_targets
            row["targets_at_95pct_coverage"] = len(passing_targets)
            row["elapsed_seconds"] = float(first_audit["elapsed_seconds"].sum())
            row["variants_per_second"] = float(
                len(first) / max(float(row["elapsed_seconds"]), 1e-12)
            )
            provenance = adapter.provenance()
            row["checkpoint_sha256"] = provenance["checkpoint_sha256"]
            row["container_digest"] = provenance["container_sha256"]
            checkpoint_complete = not specification.checkpoint or bool(row["checkpoint_sha256"])
            row["provenance_complete"] = bool(checkpoint_complete and row["container_digest"])
            row["repeat_spearman"] = float(
                spearmanr(merged["score_first"], merged["score_second"]).statistic
            )
            if parity_dir is not None:
                parity_path = Path(parity_dir) / f"{specification.model_id}.csv"
                if parity_path.is_file():
                    row["parity_spearman"] = _parity_correlation(first, pd.read_csv(parity_path))
            parity_passes = not specification.parity_required or (
                np.isfinite(row["parity_spearman"]) and float(row["parity_spearman"]) >= 0.99
            )
            eligible = (
                specification.license_status == "permitted"
                and float(row["coverage"]) >= 0.95
                and float(row["repeat_spearman"]) >= 0.999
                and parity_passes
                and bool(row["provenance_complete"])
            )
            row["execution_status"] = "passed"
            row["primary_eligible"] = eligible
            row["exclusion_reason"] = "" if eligible else "one_or_more_preflight_gates_failed"
        except Exception as exception:  # noqa: BLE001 - preflight must audit, not abort
            row["execution_status"] = "failed"
            row["exclusion_reason"] = f"{type(exception).__name__}: {exception}"
        rows.append(row)
    audit = pd.DataFrame(rows)
    eligible_ids = audit.loc[audit["primary_eligible"].astype(bool), "model_id"].astype(str)
    eligible_sets = [passing_targets_by_model.get(model_id, set()) for model_id in eligible_ids]
    shared_targets = set.intersection(*eligible_sets) if eligible_sets else set()
    family_count = int(audit.loc[audit["primary_eligible"].astype(bool), "family"].nunique())
    feasibility_passed = bool(
        len(eligible_ids) >= 8 and family_count >= 4 and len(shared_targets) >= 300
    )
    audit["primary_eligible_model_count"] = len(eligible_ids)
    audit["primary_family_count"] = family_count
    audit["primary_shared_target_count"] = len(shared_targets)
    audit["primary_shared_targets_sha256"] = (
        hashlib.sha256("\n".join(sorted(shared_targets)).encode("utf-8")).hexdigest()
        if shared_targets
        else ""
    )
    audit["feasibility_gate_passed"] = feasibility_passed
    return audit


def write_panel_predictions(
    model_config: Path,
    targets_path: Path,
    variants_path: Path,
    output_dir: Path,
    *,
    protocol_id: str,
    cache_dir: Path,
    model_ids: set[str] | None = None,
) -> dict[str, Path]:
    targets = pd.read_csv(targets_path)
    variants = pd.read_csv(variants_path)
    specifications = load_model_specifications(model_config)
    if model_ids is not None:
        specifications = [item for item in specifications if item.model_id in model_ids]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    provenance: dict[str, object] = {}
    for specification in specifications:
        adapter = adapter_from_specification(specification)
        model_predictions, audit = score_panel(
            adapter,
            targets,
            variants,
            protocol_id=protocol_id,
            cache_dir=cache_dir,
        )
        predictions.append(model_predictions)
        audits.append(audit)
        provenance[specification.model_id] = adapter.provenance()
    outputs = {
        "predictions": output_dir / "predictions.csv",
        "audit": output_dir / "prediction-audit.csv",
        "provenance": output_dir / "model-provenance.json",
    }
    write_table(pd.concat(predictions, ignore_index=True), outputs["predictions"])
    write_table(pd.concat(audits, ignore_index=True), outputs["audit"])
    outputs["provenance"].write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return outputs
