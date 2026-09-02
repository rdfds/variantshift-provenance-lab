"""Dependency-free SVG for the locked-box MaveDB external validation."""

from __future__ import annotations

from html import escape
from pathlib import Path

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
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" fill="{color}" '
        f'font-family="system-ui,sans-serif" font-size="{size}" font-weight="{weight}">'
        f"{escape(value)}</text>"
    )


def _panel(parts: list[str], x: int, width: int, title: str, subtitle: str) -> None:
    parts.append(
        f'<rect x="{x}" y="174" width="{width}" height="500" rx="18" '
        'fill="#101827" stroke="#253247"/>'
    )
    _text(parts, x + 24, 210, title, size=17, color="#f1f5f9", weight=720)
    _text(parts, x + 24, 232, subtitle, size=10)


def render_external_figure(
    protocol: dict[str, object],
    assay_audit: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    protein_metrics: pd.DataFrame,
    output: Path,
) -> Path:
    """Render cohort attrition, external estimates, and per-protein heterogeneity."""
    required_audit = {"eligible", "single_missense_variants", "protein_id"}
    required_bootstrap = {"model", "mean_spearman", "ci_lower", "ci_upper"}
    required_proteins = {"protein_id", "model", "spearman"}
    for frame, required, label in (
        (assay_audit, required_audit, "assay audit"),
        (bootstrap_summary, required_bootstrap, "bootstrap summary"),
        (protein_metrics, required_proteins, "protein metrics"),
    ):
        if missing := required.difference(frame.columns):
            raise ValueError(f"External {label} is missing: {', '.join(sorted(missing))}")

    panel = protocol["panel"]
    eligible_audit = assay_audit.loc[assay_audit["eligible"].astype(bool)]
    masked = protein_metrics.loc[protein_metrics["model"].eq("masked_marginal")].sort_values(
        "spearman", ascending=False
    )
    width, height = 1400, 720
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        ),
        '<title id="title">VariantShift locked-box external validation</title>',
        (
            '<desc id="desc">Outcome-blind cohort attrition, external ESM-2 estimates, and '
            "per-protein performance heterogeneity.</desc>"
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
        "LOCKED-BOX EXTERNAL VALIDATION · MAVEDB",
        size=13,
        color="#5eead4",
        weight=760,
    )
    _text(
        parts,
        42,
        82,
        "A real external signal survives—but it is much smaller than the benchmark suggested.",
        size=27,
        color="#f1f5f9",
        weight=760,
    )
    stats = (
        (int(panel["selected_assays"]), "locked assays"),
        (int(eligible_audit["protein_id"].nunique()), "eligible proteins"),
        (int(eligible_audit["single_missense_variants"].sum()), "assay measurements"),
        (10_000, "nested bootstraps"),
    )
    for index, (value, label) in enumerate(stats):
        x = 42 + 205 * index
        _text(parts, x, 123, f"{value:,}", size=23, color="#f8fafc", weight=760)
        _text(parts, x, 145, label, size=11)

    _panel(parts, 30, 390, "Outcome-blind cohort", "Every exclusion remains published")
    _panel(parts, 435, 430, "External performance", "Protein-balanced mean within-assay Spearman")
    _panel(parts, 880, 490, "Heterogeneity", "Masked-marginal Spearman by protein")

    funnel = (
        (2_805, "public score sets", "registry snapshot"),
        (int(panel["metadata_candidates"]), "metadata candidates", "post-v1.3 + size rules"),
        (int(panel["selected_assays"]), "locked assays", "no ProteinGym family overlap"),
        (int(eligible_audit["urn"].nunique()), "directed assays", "frozen outcome rules"),
    )
    maximum_funnel_width = 300.0
    for index, (value, label, note) in enumerate(funnel):
        y = 270 + index * 92
        visual_fraction = (value / funnel[0][0]) ** 0.35
        box_width = max(150.0, maximum_funnel_width * visual_fraction)
        x = 55 + (maximum_funnel_width - box_width) / 2
        color = "#5eead4" if index == len(funnel) - 1 else "#60a5fa"
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{box_width:.1f}" height="54" rx="12" '
            f'fill="#131f31" stroke="{color}"/>'
        )
        _text(parts, x + 15, y + 23, f"{value:,} {label}", size=13, color="#f1f5f9", weight=680)
        _text(parts, x + 15, y + 42, note, size=9)

    estimates = [
        ("ProteinGym ESM2-8M", 0.202902, None, None, "#64748b"),
    ]
    for model, label, color in (
        ("masked_marginal", "External masked", "#5eead4"),
        ("wild_type_marginal", "External WT marginal", "#a78bfa"),
    ):
        row = bootstrap_summary.loc[bootstrap_summary["model"].eq(model)]
        if len(row) != 1:
            raise ValueError(f"Expected one external bootstrap row for {model}")
        current = row.iloc[0]
        estimates.append(
            (
                label,
                float(current["mean_spearman"]),
                float(current["ci_lower"]),
                float(current["ci_upper"]),
                color,
            )
        )
    chart_left, chart_right = 485.0, 825.0
    scale = lambda value: chart_left + (value + 0.05) / 0.30 * (chart_right - chart_left)
    for tick in (-0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        x = scale(tick)
        parts.append(f'<line x1="{x:.1f}" y1="275" x2="{x:.1f}" y2="560" stroke="#253247"/>')
        _text(parts, x, 585, f"{tick:.2f}", size=9, anchor="middle")
    for index, (label, value, low, high, color) in enumerate(estimates):
        y = 315 + index * 105
        _text(parts, 470, y - 24, label, size=12, color="#cbd5e1", weight=650)
        if low is not None and high is not None:
            parts.append(
                f'<line x1="{scale(low):.1f}" y1="{y}" x2="{scale(high):.1f}" y2="{y}" '
                f'stroke="{color}" stroke-width="6" stroke-linecap="round"/>'
            )
        parts.append(
            f'<circle cx="{scale(value):.1f}" cy="{y}" r="8" fill="{color}" '
            'stroke="#f8fafc" stroke-width="1.5"/>'
        )
        _text(parts, scale(value) + 13, y + 4, f"{value:.3f}", size=12, color="#f8fafc", weight=720)
    _text(parts, 470, 624, "External top-decile recall: 0.106 (random baseline ≈ 0.100)", size=10)
    _text(parts, 470, 644, "Masked − WT marginal: 0.002 [−0.006, 0.010]", size=10)

    x_zero = 1050.0
    protein_scale = lambda value: x_zero + value / 0.35 * 245.0
    parts.append('<line x1="1050" y1="255" x2="1050" y2="640" stroke="#64748b"/>')
    for index, row in enumerate(masked.itertuples(index=False)):
        y = 278 + index * 35
        value = float(row.spearman)
        x = protein_scale(value)
        color = "#fb7185" if value < 0 else "#5eead4"
        _text(parts, 910, y + 4, str(row.protein_id), size=10, color="#cbd5e1")
        parts.append(
            f'<line x1="{x_zero:.1f}" y1="{y}" x2="{x:.1f}" y2="{y}" '
            f'stroke="{color}" stroke-width="3"/>'
        )
        parts.append(f'<circle cx="{x:.1f}" cy="{y}" r="5" fill="{color}"/>')
        _text(parts, x + (8 if value >= 0 else -8), y + 4, f"{value:.3f}", size=9, anchor="start" if value >= 0 else "end")
    _text(parts, 1050, 660, "0", size=9, anchor="middle")
    _text(parts, 1295, 660, "0.35", size=9, anchor="middle")
    parts.append("</svg>")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8")
    return output

