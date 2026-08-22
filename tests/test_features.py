import numpy as np

from variantshift.features import (
    BIOPHYSICAL_FEATURE_NAMES,
    AdditiveMutationEncoder,
    biophysical_feature_dict,
    biophysical_matrix,
)


def test_wild_type_features_are_zero() -> None:
    features = biophysical_feature_dict("")
    assert set(features) == set(BIOPHYSICAL_FEATURE_NAMES)
    assert all(value == 0 for value in features.values())


def test_property_deltas_are_additive() -> None:
    single_a = biophysical_feature_dict("A1C")
    single_b = biophysical_feature_dict("D2E")
    double = biophysical_feature_dict("A1C/D2E")
    assert double["mutation_count"] == 2
    assert double["hydropathy_delta_sum"] == (
        single_a["hydropathy_delta_sum"] + single_b["hydropathy_delta_sum"]
    )


def test_biophysical_matrix_has_stable_shape() -> None:
    matrix = biophysical_matrix(["", "A1C", "A1C/D2E"])
    assert matrix.shape == (3, len(BIOPHYSICAL_FEATURE_NAMES))
    assert np.isfinite(matrix).all()


def test_additive_encoder_handles_unseen_mutations() -> None:
    encoder = AdditiveMutationEncoder()
    train = encoder.fit_transform(["A1C", "D2E"])
    test = encoder.transform(["A1D", "D2E"])
    assert train.shape[1] == test.shape[1]

