"""Interpretable biochemical and additive mutation features."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.feature_extraction import DictVectorizer

from .mutations import Mutation, parse_variant

# Kyte-Doolittle hydropathy, approximate side-chain volume, charge at neutral pH,
# and coarse residue-class indicators. Values are intentionally transparent rather
# than learned from the evaluation dataset.
HYDROPATHY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}
VOLUME = {
    "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
    "G": 60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
    "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
    "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
}
CHARGE = {amino_acid: 0.0 for amino_acid in HYDROPATHY}
CHARGE.update({"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1})
AROMATIC = frozenset("FWY")
POLAR = frozenset("CDEHKNQRSTY")

BIOPHYSICAL_FEATURE_NAMES = (
    "mutation_count",
    "position_mean",
    "position_std",
    "position_min",
    "position_max",
    "hydropathy_delta_sum",
    "hydropathy_delta_abs",
    "volume_delta_sum",
    "volume_delta_abs",
    "charge_delta_sum",
    "charge_delta_abs",
    "aromatic_delta",
    "polar_delta",
    "proline_introduced",
    "glycine_introduced",
)


def _property_delta(mutation: Mutation, values: dict[str, float]) -> float:
    if mutation.is_stop:
        return -values[mutation.reference]
    return values[mutation.alternate] - values[mutation.reference]


def biophysical_feature_dict(codes: str) -> dict[str, float]:
    mutations = parse_variant(codes)
    if not mutations:
        return {name: 0.0 for name in BIOPHYSICAL_FEATURE_NAMES}

    positions = np.asarray([mutation.position for mutation in mutations], dtype=float)
    hydropathy = [_property_delta(mutation, HYDROPATHY) for mutation in mutations]
    volume = [_property_delta(mutation, VOLUME) for mutation in mutations]
    charge = [_property_delta(mutation, CHARGE) for mutation in mutations]

    def class_delta(mutation: Mutation, residue_class: frozenset[str]) -> float:
        before = float(mutation.reference in residue_class)
        after = float(mutation.alternate in residue_class)
        return after - before

    return {
        "mutation_count": float(len(mutations)),
        "position_mean": float(positions.mean()),
        "position_std": float(positions.std()),
        "position_min": float(positions.min()),
        "position_max": float(positions.max()),
        "hydropathy_delta_sum": float(sum(hydropathy)),
        "hydropathy_delta_abs": float(sum(map(abs, hydropathy))),
        "volume_delta_sum": float(sum(volume)),
        "volume_delta_abs": float(sum(map(abs, volume))),
        "charge_delta_sum": float(sum(charge)),
        "charge_delta_abs": float(sum(map(abs, charge))),
        "aromatic_delta": float(sum(class_delta(m, AROMATIC) for m in mutations)),
        "polar_delta": float(sum(class_delta(m, POLAR) for m in mutations)),
        "proline_introduced": float(sum(m.alternate == "P" for m in mutations)),
        "glycine_introduced": float(sum(m.alternate == "G" for m in mutations)),
    }


def additive_feature_dict(codes: str) -> dict[str, float]:
    """Add mutation identity features to the biochemical representation."""
    features = biophysical_feature_dict(codes)
    for mutation in parse_variant(codes):
        features[f"position={mutation.position}"] = 1.0
        features[f"substitution={mutation.reference}>{mutation.alternate}"] = 1.0
        features[f"mutation={mutation}"] = 1.0
    return features


def biophysical_matrix(codes: Iterable[str]) -> np.ndarray:
    rows = [biophysical_feature_dict(code) for code in codes]
    return np.asarray(
        [[row[name] for name in BIOPHYSICAL_FEATURE_NAMES] for row in rows],
        dtype=float,
    )


class AdditiveMutationEncoder:
    """Sklearn-compatible sparse encoder for position and substitution effects."""

    def __init__(self) -> None:
        self.vectorizer = DictVectorizer(sparse=True, sort=True)

    def fit_transform(self, codes: Iterable[str]):
        return self.vectorizer.fit_transform(additive_feature_dict(code) for code in codes)

    def transform(self, codes: Iterable[str]):
        return self.vectorizer.transform(additive_feature_dict(code) for code in codes)

    def feature_names(self) -> list[str]:
        return list(self.vectorizer.get_feature_names_out())

