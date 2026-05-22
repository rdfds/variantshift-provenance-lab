"""Transparent supervised baselines for variant-effect prediction."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import AdditiveMutationEncoder, biophysical_matrix


class VariantRegressor(Protocol):
    name: str

    def fit(self, codes: Iterable[str], target: np.ndarray) -> "VariantRegressor": ...

    def predict(self, codes: Iterable[str]) -> np.ndarray: ...


class MeanBaseline:
    name = "mean"

    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(self, codes: Iterable[str], target: np.ndarray) -> "MeanBaseline":
        del codes
        self.mean_ = float(np.mean(target))
        return self

    def predict(self, codes: Iterable[str]) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("Model is not fitted")
        return np.full(len(list(codes)), self.mean_, dtype=float)


class BiophysicalRidge:
    name = "biophysical_ridge"

    def __init__(self, alpha: float = 10.0) -> None:
        self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(self, codes: Iterable[str], target: np.ndarray) -> "BiophysicalRidge":
        self.model.fit(biophysical_matrix(codes), target)
        return self

    def predict(self, codes: Iterable[str]) -> np.ndarray:
        return np.asarray(self.model.predict(biophysical_matrix(codes)), dtype=float)


class AdditiveRidge:
    name = "additive_ridge"

    def __init__(self, alpha: float = 20.0) -> None:
        self.encoder = AdditiveMutationEncoder()
        self.model = Ridge(alpha=alpha)

    def fit(self, codes: Iterable[str], target: np.ndarray) -> "AdditiveRidge":
        matrix = self.encoder.fit_transform(codes)
        self.model.fit(matrix, target)
        return self

    def predict(self, codes: Iterable[str]) -> np.ndarray:
        matrix = self.encoder.transform(codes)
        return np.asarray(self.model.predict(matrix), dtype=float)


def baseline_factories() -> dict[str, type[MeanBaseline] | type[BiophysicalRidge] | type[AdditiveRidge]]:
    return {
        MeanBaseline.name: MeanBaseline,
        BiophysicalRidge.name: BiophysicalRidge,
        AdditiveRidge.name: AdditiveRidge,
    }

