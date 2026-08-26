from xml.etree import ElementTree

import pandas as pd

from variantshift.multiprotein_visualize import ESM_SCALE_ORDER, render_multiprotein_figure


def test_multiprotein_figure_is_standalone_and_semantically_labeled(tmp_path):
    assays = pd.DataFrame(
        {
            "model": ["additive_ridge"] * 3,
            "spearman_gap": [0.1, 0.2, -0.05],
            "n_seeds": [10, 10, 10],
        }
    )
    supervised = pd.DataFrame(
        [
            {
                "model": model,
                "n_assays": 3,
                "n_proteins": 2,
                "random_spearman_mean": random,
                "position_spearman_mean": position,
            }
            for model, random, position in (
                ("biophysical_ridge", 0.3, 0.2),
                ("additive_ridge", 0.6, 0.3),
            )
        ]
    )
    esm = pd.DataFrame(
        [
            {
                "model": model,
                "n_assays": 3,
                "n_proteins": 2,
                "random_spearman_mean": 0.2 + index * 0.05,
                "position_spearman_mean": 0.18 + index * 0.05,
            }
            for index, model in enumerate(ESM_SCALE_ORDER)
        ]
    )
    output = render_multiprotein_figure(assays, supervised, esm, tmp_path / "figure.svg")
    text = output.read_text()
    assert 'role="img"' in text
    assert "ESM scale audit" in text
    assert "not supervised training-distribution shift" in text
    ElementTree.fromstring(text)
