"""Biologically meaningful train/test splits with explicit leakage audits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .mutations import mutated_positions


@dataclass(frozen=True)
class VariantSplit:
    name: str
    train_indices: np.ndarray
    test_indices: np.ndarray
    excluded_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        train = set(map(int, self.train_indices))
        test = set(map(int, self.test_indices))
        if train & test:
            raise ValueError("Train and test indices must be disjoint")
        if not train or not test:
            raise ValueError("A split requires non-empty train and test sets")


def random_variant_split(
    frame: pd.DataFrame, *, test_size: float = 0.2, seed: int = 42
) -> VariantSplit:
    indices = np.arange(len(frame))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train, test = next(
        splitter.split(indices, groups=frame["mutation_codes"].astype(str).to_numpy())
    )
    return VariantSplit(
        name="random_variant",
        train_indices=np.sort(train),
        test_indices=np.sort(test),
        metadata={"test_size": test_size, "seed": seed},
    )


def position_holdout_split(
    frame: pd.DataFrame,
    *,
    position_fraction: float = 0.2,
    seed: int = 42,
) -> VariantSplit:
    """Hold out residues and discard mixed variants that would bridge the boundary."""
    position_sets = frame["mutation_codes"].map(mutated_positions)
    all_positions = sorted(set().union(*position_sets))
    if len(all_positions) < 2:
        raise ValueError("Position holdout requires at least two mutated positions")

    rng = np.random.default_rng(seed)
    shuffled = np.asarray(all_positions, dtype=int)
    rng.shuffle(shuffled)
    count = min(len(all_positions) - 1, max(1, round(len(all_positions) * position_fraction)))
    held_out = frozenset(map(int, shuffled[:count]))

    train: list[int] = []
    test: list[int] = []
    excluded: list[int] = []
    for index, positions in enumerate(position_sets):
        if not positions:
            excluded.append(index)
        elif positions.issubset(held_out):
            test.append(index)
        elif positions.isdisjoint(held_out):
            train.append(index)
        else:
            excluded.append(index)

    return VariantSplit(
        name="position_holdout",
        train_indices=np.asarray(train, dtype=int),
        test_indices=np.asarray(test, dtype=int),
        excluded_indices=np.asarray(excluded, dtype=int),
        metadata={
            "position_fraction": position_fraction,
            "seed": seed,
            "held_out_positions": sorted(held_out),
        },
    )


def modulo_position_split(
    frame: pd.DataFrame,
    *,
    fold: int,
    n_folds: int = 5,
) -> VariantSplit:
    """Hold out positions assigned by their one-indexed residue number modulo ``n_folds``.

    This mirrors ProteinGym's deterministic modulo protocol. Variants spanning train and
    test positions are excluded, although the ProteinGym single-substitution benchmark has
    no such bridging variants.
    """
    if n_folds < 2:
        raise ValueError("Modulo splitting requires at least two folds")
    if not 0 <= fold < n_folds:
        raise ValueError("Fold must lie in [0, n_folds)")
    position_sets = frame["mutation_codes"].map(mutated_positions)
    all_positions = sorted(set().union(*position_sets))
    held_out = frozenset(position for position in all_positions if position % n_folds == fold)
    if not held_out or held_out == frozenset(all_positions):
        raise ValueError("Modulo split produced an empty train or test position set")
    return _split_on_held_out_positions(
        position_sets,
        held_out,
        name="modulo_position",
        metadata={"fold": fold, "n_folds": n_folds},
    )


def contiguous_position_split(
    frame: pd.DataFrame,
    *,
    fold: int,
    n_folds: int = 5,
) -> VariantSplit:
    """Hold out one contiguous block of observed residue positions.

    Blocks contain nearly equal numbers of *observed mutated positions*, matching the
    ProteinGym definition rather than dividing the full sequence coordinate range.
    """
    if n_folds < 2:
        raise ValueError("Contiguous splitting requires at least two folds")
    if not 0 <= fold < n_folds:
        raise ValueError("Fold must lie in [0, n_folds)")
    position_sets = frame["mutation_codes"].map(mutated_positions)
    all_positions = np.asarray(sorted(set().union(*position_sets)), dtype=int)
    if len(all_positions) < n_folds:
        raise ValueError("Contiguous splitting requires at least one position per fold")
    held_out = frozenset(map(int, np.array_split(all_positions, n_folds)[fold]))
    return _split_on_held_out_positions(
        position_sets,
        held_out,
        name="contiguous_position",
        metadata={"fold": fold, "n_folds": n_folds},
    )


def _split_on_held_out_positions(
    position_sets: pd.Series,
    held_out: frozenset[int],
    *,
    name: str,
    metadata: dict[str, Any],
) -> VariantSplit:
    train: list[int] = []
    test: list[int] = []
    excluded: list[int] = []
    for index, positions in enumerate(position_sets):
        if not positions:
            excluded.append(index)
        elif positions.issubset(held_out):
            test.append(index)
        elif positions.isdisjoint(held_out):
            train.append(index)
        else:
            excluded.append(index)
    return VariantSplit(
        name=name,
        train_indices=np.asarray(train, dtype=int),
        test_indices=np.asarray(test, dtype=int),
        excluded_indices=np.asarray(excluded, dtype=int),
        metadata={**metadata, "held_out_positions": sorted(held_out)},
    )


def mutation_depth_split(
    frame: pd.DataFrame,
    *,
    max_train_depth: int = 1,
    max_test_depth: int = 5,
) -> VariantSplit:
    """Train on shallow variants and test combinatorial extrapolation."""
    depth = frame["goi_amino_mutations"].to_numpy(dtype=int)
    train = np.flatnonzero((depth >= 1) & (depth <= max_train_depth))
    test = np.flatnonzero((depth > max_train_depth) & (depth <= max_test_depth))
    included = set(map(int, train)) | set(map(int, test))
    excluded = np.asarray([i for i in range(len(frame)) if i not in included], dtype=int)
    return VariantSplit(
        name="mutation_depth",
        train_indices=train,
        test_indices=test,
        excluded_indices=excluded,
        metadata={
            "max_train_depth": max_train_depth,
            "max_test_depth": max_test_depth,
        },
    )


def leakage_audit(frame: pd.DataFrame, split: VariantSplit) -> dict[str, Any]:
    train_codes = set(frame.iloc[split.train_indices]["mutation_codes"])
    test_codes = set(frame.iloc[split.test_indices]["mutation_codes"])
    train_positions = set().union(
        *frame.iloc[split.train_indices]["mutation_codes"].map(mutated_positions)
    )
    test_positions = set().union(
        *frame.iloc[split.test_indices]["mutation_codes"].map(mutated_positions)
    )
    shared_positions = sorted(train_positions & test_positions)
    return {
        "train_rows": len(split.train_indices),
        "test_rows": len(split.test_indices),
        "excluded_rows": len(split.excluded_indices),
        "exact_variant_overlap": len(train_codes & test_codes),
        "train_positions": len(train_positions),
        "test_positions": len(test_positions),
        "shared_position_count": len(shared_positions),
        "shared_positions": shared_positions,
    }
