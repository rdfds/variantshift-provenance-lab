"""Dependency-free SVG summary for the structured-shift extension."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


def _text(
    parts: list[str],
    x: float,
    y: float,
    value: str,
    *,
    size: int = 11,
    color: str = "#94a3b8",
    weight: int = 400,
    anchor: str = "start",
) -> None:
    parts.append(
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" '
        f'fill="{color}" font-family="system-ui,sans-serif" font-size="{size}" '
        f'font-weight="{weight}">{escape(value)}</text>'
    )


def _panel(parts: list[str], x: int, width: int, title: str, subtitle: str) -> None:
    parts.append(
        f'<rect x="{x}" y="176" width="{width}" height="486" rx="18" '
        'fill="#101827" stroke="#253247" filter="url(#shadow)"/>'
    )
    _text(parts, x + 28, 215, title, size=18, color="#f1f5f9", weight=720)
    _text(parts, x + 28, 238, subtitle, size=11)


def render_extended_figure(
    official: pd.DataFrame,
    probe: pd.DataFrame,
    heldout: pd.DataFrame,
    crossover: pd.DataFrame,
    output: Path,
) -> Path:
    """Render model, calibration, and transfer results in one audited figure."""
    requirements = (
        (
            official,
            {"model", "split", "mean_spearman", "n_assays", "n_proteins"},
            "official",
        ),
        (
            probe,
            {
                "split",
                "calibration_method",
                "mean_spearman",
                "mean_observed_coverage",
                "mean_normalized_mean_interval_width",
            },
            "probe",
        ),
        (
            heldout,
            {"model", "calibration_method", "mean_spearman", "mean_observed_coverage"},
            "held-out protein",
        ),
        (
            crossover,
            {"model", "roc_auc", "accuracy", "majority_accuracy", "n_examples"},
            "crossover",
        ),
    )
    missing = {
        f"{label}:{column}"
        for frame, columns, label in requirements
        for column in columns.difference(frame.columns)
    }
    if missing:
        raise ValueError(f"Extended figure input is missing: {', '.join(sorted(missing))}")

    official_rows = official.loc[
        official["split"].isin(["random_variant", "contiguous_position"])
    ].copy()
    if len(official_rows) != 6:
        raise ValueError("Expected three official models under two split protocols")
    probe_rows = probe.loc[
        probe["calibration_method"].eq("standard_split")
        & probe["split"].isin(["random_variant", "contiguous_position"])
    ]
    if len(probe_rows) != 2:
        raise ValueError("Expected two local-probe split summaries")

    width, height = 1320, 720
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        ),
        '<title id="title">VariantShift structured-shift extension</title>',
        '<desc id="desc">Modern supervised baselines, calibration diagnostics, held-out-protein transfer, and model-crossover prediction.</desc>',
        (
            '<defs><linearGradient id="background" x1="0" y1="0" x2="1" y2="1">'
            '<stop stop-color="#071a2d"/><stop offset="1" stop-color="#08111f"/>'
            '</linearGradient><filter id="shadow"><feDropShadow dx="0" dy="7" '
            'stdDeviation="9" flood-opacity=".25"/></filter></defs>'
        ),
        f'<rect width="{width}" height="{height}" rx="26" fill="url(#background)"/>',
    ]
    _text(
        parts,
        48,
        48,
        "STRUCTURED SHIFT · PROTEINGYM v1.3",
        size=13,
        color="#5eead4",
        weight=760,
    )
    _text(
        parts,
        48,
        86,
        "Harder splits change model rankings—and expose calibration failure.",
        size=29,
        color="#f1f5f9",
        weight=760,
    )
    stats = (
        (str(int(official["n_assays"].max())), "audited assays"),
        (str(int(official["n_proteins"].max())), "unique proteins"),
        (str(int(crossover["n_examples"].max())), "held-out crossover decisions"),
    )
    for index, (value, label) in enumerate(stats):
        x = 48 + 210 * index
        _text(parts, x, 127, value, size=24, color="#f8fafc", weight=760)
        _text(parts, x, 149, label, size=12)

    _panel(parts, 38, 410, "Modern supervised baselines", "Mean Spearman · UniProt-weighted")
    _panel(parts, 465, 410, "Intervals under shift", "Local ESM-2 probe · nominal coverage 0.80")
    _panel(parts, 892, 390, "Cross-protein transfer", "Proteins absent from corresponding training folds")

    model_order = (
        ("esm1v_embedding_probe", "ESM-1v probe"),
        ("protein_npt", "ProteinNPT"),
        ("kermut", "Kermut"),
        ("local_probe", "Local ESM-2 probe"),
    )
    model_values: dict[str, dict[str, float]] = {}
    for model, _ in model_order[:-1]:
        rows = official_rows.loc[official_rows["model"].eq(model)]
        model_values[model] = dict(zip(rows["split"], rows["mean_spearman"], strict=True))
    model_values["local_probe"] = dict(
        zip(probe_rows["split"], probe_rows["mean_spearman"], strict=True)
    )
    x0, max_width = 180.0, 235.0
    for index, (model, label) in enumerate(model_order):
        y = 286 + index * 84
        _text(parts, 66, y - 13, label, size=12, color="#cbd5e1", weight=650)
        for offset, split, color, split_label in (
            (0, "random_variant", "#5eead4", "random"),
            (26, "contiguous_position", "#a78bfa", "contiguous"),
        ):
            value = float(model_values[model][split])
            bar_width = max_width * value / 0.85
            parts.append(
                f'<rect x="{x0:.1f}" y="{y + offset - 28:.1f}" width="{bar_width:.1f}" '
                f'height="16" rx="8" fill="{color}"/>'
            )
            _text(parts, x0 + bar_width + 7, y + offset - 15, f"{value:.3f}", size=10)
            _text(parts, 66, y + offset + 4, split_label, size=9, color="#64748b")

    calibration = probe.loc[
        probe["split"].isin(["random_variant", "position_holdout", "contiguous_position"])
    ].copy()
    methods = (
        ("standard_split", "Standard", "#5eead4"),
        ("mondrian_substitution", "Group-aware", "#60a5fa"),
        ("position_distance_scaled", "Distance-scaled", "#fb7185"),
    )
    split_order = (
        ("random_variant", "Random"),
        ("position_holdout", "Random position"),
        ("contiguous_position", "Contiguous"),
    )
    chart_left, chart_top, chart_width, chart_height = 512.0, 292.0, 310.0, 245.0
    y_scale = lambda value: chart_top + chart_height * (1.0 - value)
    for tick in (0.4, 0.6, 0.8, 1.0):
        y = y_scale(tick)
        parts.append(
            f'<line x1="{chart_left:.1f}" y1="{y:.1f}" x2="{chart_left + chart_width:.1f}" '
            f'y2="{y:.1f}" stroke="{("#fb7185" if tick == 0.8 else "#253247")}" '
            f'stroke-dasharray="{("5 4" if tick == 0.8 else "0")}"/>'
        )
        _text(parts, chart_left - 9, y + 4, f"{tick:.1f}", size=9, anchor="end")
    group_x = np.linspace(chart_left + 42, chart_left + chart_width - 42, 3)
    for split_index, (split, label) in enumerate(split_order):
        _text(parts, group_x[split_index], 562, label, size=9, anchor="middle")
        rows = calibration.loc[calibration["split"].eq(split)].set_index(
            "calibration_method"
        )
        for method_index, (method, _, color) in enumerate(methods):
            value = float(rows.loc[method, "mean_observed_coverage"])
            x = group_x[split_index] + (method_index - 1) * 17
            y = y_scale(value)
            parts.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}">'
                f'<title>{escape(label)} · {escape(method)}: coverage {value:.3f}</title></circle>'
            )
    for index, (_, label, color) in enumerate(methods):
        x = 502 + index * 116
        parts.append(f'<circle cx="{x}" cy="600" r="4" fill="{color}"/>')
        _text(parts, x + 9, 604, label, size=9)
    shifted = calibration.loc[
        calibration["split"].eq("contiguous_position")
        & calibration["calibration_method"].eq("position_distance_scaled")
    ].iloc[0]
    _text(
        parts,
        505,
        632,
        f'Distance scaling: {shifted["mean_observed_coverage"]:.3f} coverage, '
        f'{shifted["mean_normalized_mean_interval_width"]:.2f}× normalized width',
        size=10,
        color="#cbd5e1",
    )

    heldout_standard = heldout.loc[
        heldout["calibration_method"].eq("standard_split")
    ].set_index("model")
    heldout_models = (
        ("cross_protein_ridge", "Ridge"),
        ("cross_protein_histgb", "Nonlinear HistGB"),
    )
    for index, (model, label) in enumerate(heldout_models):
        y = 300 + index * 86
        value = float(heldout_standard.loc[model, "mean_spearman"])
        _text(parts, 920, y - 22, label, size=12, color="#cbd5e1", weight=650)
        parts.append(
            f'<rect x="920" y="{y - 10}" width="{value * 470:.1f}" height="18" '
            'rx="9" fill="#5eead4"/>'
        )
        _text(parts, 920 + value * 470 + 8, y + 4, f"{value:.3f}", size=11)
    logistic = crossover.loc[crossover["model"].eq("logistic")].iloc[0]
    _text(parts, 920, 466, "Crossover predictor", size=12, color="#cbd5e1", weight=650)
    for index, (label, value, color) in enumerate(
        (
            ("ROC-AUC", float(logistic["roc_auc"]), "#a78bfa"),
            ("accuracy", float(logistic["accuracy"]), "#60a5fa"),
            ("majority baseline", float(logistic["majority_accuracy"]), "#64748b"),
        )
    ):
        y = 500 + index * 42
        _text(parts, 920, y + 4, label, size=10)
        parts.append(
            f'<rect x="1022" y="{y - 10}" width="{value * 220:.1f}" height="16" '
            f'rx="8" fill="{color}"/>'
        )
        _text(parts, 1027 + value * 220, y + 3, f"{value:.3f}", size=10)
    _text(
        parts,
        48,
        698,
        "Official models use ProteinGym out-of-fold predictions · local intervals are empirical under shift · held-out-protein is not held-out-family transfer",
        size=11,
        color="#64748b",
    )
    parts.append("</svg>")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8")
    return output
