"""Dependency-free SVG summary for the ProteinGym validation study."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

ESM_SCALE_ORDER = (
    "ESM2_8M",
    "ESM2_35M",
    "ESM2_150M",
    "ESM2_650M",
    "ESM2_3B",
    "ESM2_15B",
)


def _line(parts: list[str], text: str) -> None:
    parts.append(text)


def render_multiprotein_figure(
    supervised_assays: pd.DataFrame,
    supervised_aggregate: pd.DataFrame,
    esm_aggregate: pd.DataFrame,
    output: Path,
) -> Path:
    """Render aggregate model performance and assay-level shift heterogeneity."""
    supervised_required = {"model", "spearman_gap"}
    aggregate_required = {
        "model",
        "n_assays",
        "n_proteins",
        "random_spearman_mean",
        "position_spearman_mean",
    }
    missing = (
        supervised_required.difference(supervised_assays.columns)
        | aggregate_required.difference(supervised_aggregate.columns)
        | aggregate_required.difference(esm_aggregate.columns)
    )
    if missing:
        raise ValueError(f"Multi-protein figure input is missing: {', '.join(sorted(missing))}")

    additive = supervised_aggregate.loc[
        supervised_aggregate["model"].eq("additive_ridge")
    ]
    if len(additive) != 1:
        raise ValueError("Expected one additive-ridge aggregate row")
    additive_row = additive.iloc[0]
    assay_gaps = supervised_assays.loc[
        supervised_assays["model"].eq("additive_ridge"), "spearman_gap"
    ].to_numpy(dtype=float)
    if assay_gaps.size == 0:
        raise ValueError("No additive-ridge assay gaps were provided")

    width, height = 1280, 720
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        ),
        '<title id="title">VariantShift multi-protein validation</title>',
        '<desc id="desc">Supervised generalization across ProteinGym proteins and audited ESM zero-shot subset performance.</desc>',
        "<defs>",
        '<linearGradient id="background" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#071a2d"/><stop offset="1" stop-color="#08111f"/></linearGradient>',
        '<filter id="shadow"><feDropShadow dx="0" dy="7" stdDeviation="9" flood-opacity=".25"/></filter>',
        "</defs>",
        f'<rect width="{width}" height="{height}" rx="26" fill="url(#background)"/>',
        '<text x="48" y="48" fill="#5eead4" font-family="system-ui,sans-serif" font-size="13" font-weight="760" letter-spacing="2">EXTERNAL VALIDATION · PROTEINGYM v1.3</text>',
        '<text x="48" y="86" fill="#f1f5f9" font-family="system-ui,sans-serif" font-size="29" font-weight="760">Random splits overstate supervised performance across proteins.</text>',
    ]
    stats = (
        (int(additive_row["n_assays"]), "eligible assays"),
        (int(additive_row["n_proteins"]), "unique proteins"),
        (int(supervised_assays["n_seeds"].max()), "paired split seeds"),
    )
    for index, (value, label) in enumerate(stats):
        x = 48 + index * 180
        _line(
            parts,
            f'<text x="{x}" y="127" fill="#f8fafc" font-family="system-ui,sans-serif" font-size="24" font-weight="760">{value}</text>',
        )
        _line(
            parts,
            f'<text x="{x}" y="148" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="12">{escape(label)}</text>',
        )

    panels = ((38, 176, 390, 486), (445, 176, 385, 486), (847, 176, 395, 486))
    for x, y, panel_width, panel_height in panels:
        _line(
            parts,
            f'<rect x="{x}" y="{y}" width="{panel_width}" height="{panel_height}" rx="18" fill="#101827" stroke="#253247" filter="url(#shadow)"/>',
        )

    _line(parts, '<text x="66" y="215" fill="#f1f5f9" font-family="system-ui,sans-serif" font-size="18" font-weight="720">Supervised shift</text>')
    _line(parts, '<text x="66" y="238" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="12">Mean Spearman after UniProt aggregation</text>')
    supervised_models = ("biophysical_ridge", "additive_ridge")
    labels = {"biophysical_ridge": "Biophysical ridge", "additive_ridge": "Additive ridge"}
    y_positions = (330.0, 475.0)
    for model, y in zip(supervised_models, y_positions, strict=True):
        row = supervised_aggregate.loc[supervised_aggregate["model"].eq(model)]
        if row.empty:
            continue
        random_value = float(row["random_spearman_mean"].iat[0])
        position_value = float(row["position_spearman_mean"].iat[0])
        x0, available = 80.0, 315.0
        _line(parts, f'<text x="66" y="{y - 55:.0f}" fill="#cbd5e1" font-family="system-ui,sans-serif" font-size="13" font-weight="650">{escape(labels[model])}</text>')
        for offset, value, color, label in (
            (0, random_value, "#5eead4", "random"),
            (34, position_value, "#a78bfa", "unseen position"),
        ):
            bar_width = max(0.0, available * value)
            _line(parts, f'<rect x="{x0}" y="{y + offset - 28:.0f}" width="{bar_width:.1f}" height="18" rx="9" fill="{color}"/>')
            _line(parts, f'<text x="{x0 + bar_width + 8:.1f}" y="{y + offset - 14:.0f}" fill="#e2e8f0" font-family="system-ui,sans-serif" font-size="11">{value:.3f}</text>')
            _line(parts, f'<text x="{x0}" y="{y + offset + 5:.0f}" fill="#64748b" font-family="system-ui,sans-serif" font-size="9">{escape(label)}</text>')

    _line(parts, '<text x="473" y="215" fill="#f1f5f9" font-family="system-ui,sans-serif" font-size="18" font-weight="720">Assay heterogeneity</text>')
    _line(parts, '<text x="473" y="238" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="12">Additive-ridge random − position Spearman</text>')
    lower = min(-0.3, float(np.quantile(assay_gaps, 0.01)))
    upper = max(0.7, float(np.quantile(assay_gaps, 0.99)))
    x0, x1 = 480.0, 798.0
    scale = lambda value: x0 + (x1 - x0) * (value - lower) / (upper - lower)
    zero_x = scale(0.0)
    _line(parts, f'<line x1="{zero_x:.1f}" y1="268" x2="{zero_x:.1f}" y2="604" stroke="#64748b" stroke-dasharray="4 5"/>')
    sorted_gaps = np.sort(assay_gaps)
    for index, value in enumerate(sorted_gaps):
        y = 275 + 320 * index / max(1, len(sorted_gaps) - 1)
        _line(parts, f'<circle cx="{scale(float(value)):.1f}" cy="{y:.1f}" r="2.3" fill="#93c5fd" opacity=".58"/>')
    mean_gap = float(assay_gaps.mean())
    median_gap = float(np.median(assay_gaps))
    _line(parts, f'<line x1="{scale(mean_gap):.1f}" y1="265" x2="{scale(mean_gap):.1f}" y2="610" stroke="#fb7185" stroke-width="3"/>')
    _line(parts, f'<text x="473" y="632" fill="#e2e8f0" font-family="system-ui,sans-serif" font-size="11">mean {mean_gap:.3f} · median {median_gap:.3f} · each dot is one assay</text>')

    _line(parts, '<text x="875" y="215" fill="#f1f5f9" font-family="system-ui,sans-serif" font-size="18" font-weight="720">ESM scale audit</text>')
    _line(parts, '<text x="875" y="238" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="12">Official fixed zero-shot scores</text>')
    esm_rows = esm_aggregate.set_index("model")
    x_values = np.linspace(882.0, 1208.0, len(ESM_SCALE_ORDER))
    y_top, y_bottom = 282.0, 570.0
    y_scale = lambda value: y_bottom - (y_bottom - y_top) * (value + 0.05) / 0.75
    for y_tick in (0.0, 0.2, 0.4, 0.6):
        y = y_scale(y_tick)
        _line(parts, f'<line x1="875" y1="{y:.1f}" x2="1218" y2="{y:.1f}" stroke="#253247"/>')
        _line(parts, f'<text x="868" y="{y + 4:.1f}" text-anchor="end" fill="#64748b" font-family="system-ui,sans-serif" font-size="9">{y_tick:.1f}</text>')
    for metric, color in (("random_spearman_mean", "#5eead4"), ("position_spearman_mean", "#a78bfa")):
        points = []
        for model, x in zip(ESM_SCALE_ORDER, x_values, strict=True):
            value = float(esm_rows.loc[model, metric])
            y = y_scale(value)
            points.append(f"{x:.1f},{y:.1f}")
            _line(parts, f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>')
        _line(parts, f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    for model, x in zip(ESM_SCALE_ORDER, x_values, strict=True):
        label = model.replace("ESM2_", "")
        _line(parts, f'<text x="{x:.1f}" y="595" text-anchor="middle" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="9">{escape(label)}</text>')
    _line(parts, '<circle cx="890" cy="626" r="4" fill="#5eead4"/><text x="901" y="630" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="10">random subset</text>')
    _line(parts, '<circle cx="1000" cy="626" r="4" fill="#a78bfa"/><text x="1011" y="630" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="10">unseen-position subset</text>')
    _line(parts, '<text x="48" y="696" fill="#64748b" font-family="system-ui,sans-serif" font-size="11">ESM subset differences describe test composition, not supervised training-distribution shift · all intervals bootstrap UniProt IDs</text>')
    parts.append("</svg>")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8")
    return output
