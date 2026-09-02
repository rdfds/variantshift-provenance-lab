import pandas as pd

from variantshift.external_visualize import render_external_figure


def test_external_figure_renders_complete_svg(tmp_path) -> None:
    protocol = {
        "panel": {
            "selected_assays": 2,
            "metadata_candidates": 3,
        }
    }
    audit = pd.DataFrame(
        {
            "urn": ["a", "b"],
            "eligible": [True, False],
            "single_missense_variants": [600, 700],
            "protein_id": ["P1", "P2"],
        }
    )
    bootstrap = pd.DataFrame(
        {
            "model": ["masked_marginal", "wild_type_marginal"],
            "mean_spearman": [0.12, 0.10],
            "ci_lower": [0.02, 0.01],
            "ci_upper": [0.20, 0.19],
        }
    )
    proteins = pd.DataFrame(
        {
            "protein_id": ["P1", "P1"],
            "model": ["masked_marginal", "wild_type_marginal"],
            "spearman": [0.12, 0.10],
        }
    )
    output = render_external_figure(protocol, audit, bootstrap, proteins, tmp_path / "figure.svg")
    text = output.read_text()
    assert text.startswith("<svg")
    assert "Outcome-blind cohort" in text
    assert text.endswith("</svg>")
