"""Dependency-free SVG for the publication-oriented validation layers."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_LABELS = {
    "venusrem": "VenusREM",
    "prosst_2048": "ProSST",
    "s3f_msa": "S3F-MSA",
    "esm3_open": "ESM3",
    "saprot_650m": "SaProt",
    "gemme": "GEMME",
    "esm2_650m": "ESM-2 650M",
    "esmc_600m": "ESM-C 600M",
    "tranception_l": "Tranception-L",
    "siterm": "SiteRM",
    "progen3_3b": "ProGen3 3B",
    "xtrimopglm_100b": "xTrimoPGLM 100B",
}


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
        f'<rect x="{x}" y="174" width="{width}" height="500" rx="18" '
        'fill="#101827" stroke="#253247"/>'
    )
    _text(parts, x + 24, 210, title, size=17, color="#f1f5f9", weight=720)
    _text(parts, x + 24, 232, subtitle, size=10)


def _standard_spearman(summary: pd.DataFrame, model: str) -> float:
    row = summary.loc[
        summary["model"].eq(model) & summary["calibration_method"].eq("standard_split")
    ]
    if len(row) != 1:
        raise ValueError(f"Expected one standard-split summary row for {model}")
    return float(row.iloc[0]["mean_spearman"])


def render_research_figure(
    modern: pd.DataFrame,
    sequence_audit: pd.DataFrame,
    structure_audit: pd.DataFrame,
    curated_audit: pd.DataFrame,
    heldout_protein: pd.DataFrame,
    heldout_family: pd.DataFrame,
    heldout_structure: pd.DataFrame,
    heldout_curated: pd.DataFrame,
    output: Path,
) -> Path:
    """Render paired model ranking, homology audit, and grouped transfer results."""
    modern_required = {
        "model",
        "n_assays",
        "n_proteins",
        "mean_spearman",
        "spearman_ci_low",
        "spearman_ci_high",
    }
    if missing := modern_required.difference(modern.columns):
        raise ValueError(f"Modern zero-shot summary is missing: {', '.join(sorted(missing))}")
    for frame, count_column, label in (
        (sequence_audit, "n_families", "sequence"),
        (structure_audit, "n_families", "structure"),
        (curated_audit, "curated_family_count", "curated"),
    ):
        if count_column not in frame:
            raise ValueError(f"{label.title()} audit is missing {count_column}")

    width, height = 1400, 720
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        ),
        '<title id="title">VariantShift publication-grade validation</title>',
        (
            '<desc id="desc">Paired modern zero-shot rankings, independent family grouping '
            "audits, and repeated grouped transfer estimates.</desc>"
        ),
        (
            '<defs><linearGradient id="background" x1="0" y1="0" x2="1" y2="1">'
            '<stop stop-color="#071a2d"/><stop offset="1" stop-color="#08111f"/>'
            "</linearGradient></defs>"
        ),
        f'<rect width="{width}" height="{height}" rx="26" fill="url(#background)"/>',
    ]
    _text(
        parts,
        42,
        45,
        "INDEPENDENT VALIDATION · PROTEINGYM v1.3",
        size=13,
        color="#5eead4",
        weight=760,
    )
    _text(
        parts,
        42,
        82,
        "Modern model rankings and cross-family transfer survive stricter audits.",
        size=28,
        color="#f1f5f9",
        weight=760,
    )
    stats = (
        (int(modern["n_assays"].max()), "paired assays"),
        (int(modern["n_proteins"].max()), "unique proteins"),
        (len(modern), "zero-shot models"),
        (5, "grouped split repeats"),
    )
    for index, (value, label) in enumerate(stats):
        x = 42 + 175 * index
        _text(parts, x, 123, str(value), size=23, color="#f8fafc", weight=760)
        _text(parts, x, 145, label, size=11)

    _panel(
        parts,
        30,
        445,
        "Modern zero-shot landscape",
        "Exactly paired variants · protein-balanced Spearman",
    )
    _panel(
        parts,
        490,
        400,
        "Independent family audit",
        "Connected components become progressively stricter",
    )
    _panel(
        parts,
        905,
        465,
        "Repeated grouped transfer",
        "Mean within-assay Spearman · five shuffled group partitions",
    )

    ranked = modern.sort_values("mean_spearman", ascending=False).reset_index(drop=True)
    bar_left, bar_width = 137.0, 285.0
    maximum = max(0.60, float(ranked["spearman_ci_high"].max()))
    for index, row in ranked.iterrows():
        y = 266 + index * 30
        value = float(row["mean_spearman"])
        low = float(row["spearman_ci_low"])
        high = float(row["spearman_ci_high"])
        color = (
            "#5eead4" if index < 2 else ("#a78bfa" if row["model"] == "esm2_650m" else "#60a5fa")
        )
        label = MODEL_LABELS.get(str(row["model"]), str(row["model"]))
        _text(parts, 48, y + 3, label, size=9, color="#cbd5e1")
        parts.append(
            f'<rect x="{bar_left:.1f}" y="{y - 9:.1f}" width="{value / maximum * bar_width:.1f}" '
            f'height="12" rx="6" fill="{color}"/>'
        )
        low_x = bar_left + low / maximum * bar_width
        high_x = bar_left + high / maximum * bar_width
        parts.append(
            f'<line x1="{low_x:.1f}" y1="{y - 3:.1f}" x2="{high_x:.1f}" y2="{y - 3:.1f}" '
            'stroke="#f8fafc" stroke-width="1.2"/>'
        )
        _text(parts, bar_left + value / maximum * bar_width + 6, y + 2, f"{value:.3f}", size=9)

    sequence = sequence_audit.iloc[0]
    structure = structure_audit.iloc[0]
    primary_curated = curated_audit.loc[curated_audit["method"].eq("pfam_family")].iloc[0]
    clan = curated_audit.loc[curated_audit["method"].eq("pfam_clan_sensitivity")].iloc[0]
    stages = (
        (
            "MMseqs2 sequence",
            int(sequence["n_families"]),
            int(sequence["proteins_in_multi_protein_families"]),
            "≥30% identity · ≥80% reciprocal coverage",
            "#60a5fa",
        ),
        (
            "+ reciprocal Foldseek",
            int(structure["n_families"]),
            int(structure["proteins_in_multi_protein_families"]),
            "prob ≥.95 · TM ≥.50 · coverage ≥.80",
            "#a78bfa",
        ),
        (
            "+ curated Pfam family",
            int(primary_curated["curated_family_count"]),
            int(primary_curated["proteins_in_multi_protein_families"]),
            f"Pfam {primary_curated['pfam_version']} · assayed-region overlap",
            "#5eead4",
        ),
    )
    for index, (label, families, proteins, rule, color) in enumerate(stages):
        y = 274 + index * 118
        parts.append(
            f'<rect x="516" y="{y - 24}" width="348" height="88" rx="13" '
            f'fill="#131f31" stroke="{color}"/>'
        )
        _text(parts, 536, y, label, size=12, color="#f1f5f9", weight=680)
        _text(parts, 536, y + 28, f"{families} families", size=21, color=color, weight=760)
        _text(parts, 676, y + 28, f"{proteins} proteins in multi-protein groups", size=9)
        _text(parts, 536, y + 49, rule, size=9, color="#94a3b8")
        if index < 2:
            _text(parts, 690, y + 86, "↓ union homologous components", size=9, anchor="middle")
    _text(
        parts,
        520,
        628,
        f"Stress test: Pfam clans → {int(clan['curated_family_count'])} families; reported separately",
        size=10,
        color="#fb7185",
    )

    protocols = (
        ("protein", heldout_protein),
        ("sequence", heldout_family),
        ("seq+structure", heldout_structure),
        ("+Pfam", heldout_curated),
    )
    models = (
        ("cross_protein_histgb", "Nonlinear HistGB", "#5eead4"),
        ("cross_protein_ridge", "Ridge", "#a78bfa"),
    )
    chart_left, chart_right = 956.0, 1328.0
    chart_top, chart_bottom = 290.0, 545.0
    minimum, maximum_transfer = 0.44, 0.58
    scale_y = lambda value: (
        chart_bottom - (value - minimum) / (maximum_transfer - minimum) * (chart_bottom - chart_top)
    )
    x_values = np.linspace(chart_left + 25, chart_right - 25, len(protocols))
    for tick in np.arange(minimum, maximum_transfer + 0.001, 0.02):
        y = scale_y(float(tick))
        parts.append(
            f'<line x1="{chart_left:.1f}" y1="{y:.1f}" x2="{chart_right:.1f}" y2="{y:.1f}" '
            'stroke="#253247"/>'
        )
        _text(parts, chart_left - 9, y + 4, f"{tick:.2f}", size=9, anchor="end")
    for model, label, color in models:
        values = [_standard_spearman(summary, model) for _, summary in protocols]
        points = " ".join(
            f"{x:.1f},{scale_y(value):.1f}" for x, value in zip(x_values, values, strict=True)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        for x, value in zip(x_values, values, strict=True):
            y = scale_y(value)
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}"/>')
            _text(parts, x, y - 10, f"{value:.3f}", size=9, color="#f8fafc", anchor="middle")
        legend_x = 965 if model.endswith("histgb") else 1120
        parts.append(f'<circle cx="{legend_x}" cy="602" r="5" fill="{color}"/>')
        _text(parts, legend_x + 10, 606, label, size=10, color="#cbd5e1")
    for x, (label, _) in zip(x_values, protocols, strict=True):
        _text(parts, x, 570, label, size=9, anchor="middle")
    _text(
        parts,
        930,
        641,
        "Fit, calibration, and test groups are disjoint in every fold; repeat estimates are published.",
        size=9,
        color="#94a3b8",
    )
    parts.append("</svg>")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8")
    return output
