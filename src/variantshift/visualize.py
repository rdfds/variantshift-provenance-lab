"""Dependency-free SVG visualizations for committed shift analyses."""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


def _condition_key(condition: str) -> tuple[int, str]:
    suffix = str(condition).rsplit("S", maxsplit=1)[-1]
    return (int(suffix), str(condition)) if suffix.isdigit() else (10**9, str(condition))


def _condition_label(condition: str) -> str:
    return str(condition).replace("mean_y_", "")


def _color(value: float, *, lower: float, upper: float) -> str:
    fraction = float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))
    start = np.asarray([23, 37, 84], dtype=float)
    end = np.asarray([94, 234, 212], dtype=float)
    red, green, blue = np.rint(start + fraction * (end - start)).astype(int)
    return f"#{red:02x}{green:02x}{blue:02x}"


def render_shift_figure(
    gaps: pd.DataFrame,
    transfer: pd.DataFrame,
    output: Path,
    *,
    model: str = "additive_ridge",
    transfer_split: str = "position_holdout",
) -> Path:
    gap_required = {"seed", "target", "model", "spearman_gap"}
    transfer_required = {
        "split",
        "model",
        "source_condition",
        "target_condition",
        "transfer_spearman",
        "exact_variant_overlap",
    }
    missing_gap = gap_required.difference(gaps.columns)
    missing_transfer = transfer_required.difference(transfer.columns)
    if missing_gap or missing_transfer:
        missing = sorted(missing_gap | missing_transfer)
        raise ValueError(f"Shift figure input is missing columns: {', '.join(missing)}")

    gap_rows = gaps.loc[gaps["model"].eq(model)].copy()
    transfer_rows = transfer.loc[
        transfer["model"].eq(model) & transfer["split"].eq(transfer_split)
    ].copy()
    if gap_rows.empty or transfer_rows.empty:
        raise ValueError("Shift figure has no rows for the requested model and split")

    targets = sorted(gap_rows["target"].unique())
    conditions = sorted(transfer_rows["source_condition"].unique(), key=_condition_key)
    matrix = transfer_rows.pivot(
        index="source_condition",
        columns="target_condition",
        values="transfer_spearman",
    ).reindex(index=conditions, columns=conditions)
    if matrix.isna().any().any():
        raise ValueError("Condition-transfer matrix is incomplete")
    transfer_min = float(np.floor(matrix.to_numpy(dtype=float).min() * 10) / 10)
    transfer_max = float(np.ceil(matrix.to_numpy(dtype=float).max() * 10) / 10)
    if transfer_max <= transfer_min:
        transfer_max = transfer_min + 0.1

    width, height = 1180, 620
    parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        ),
        '<title id="title">VariantShift robustness and condition-transfer analysis</title>',
        '<desc id="desc">Repeated-split generalization gaps and a twenty-condition transfer matrix under unseen-position evaluation.</desc>',
        "<defs>",
        '<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">',
        '<stop stop-color="#102a43"/><stop offset="1" stop-color="#07111f"/>',
        "</linearGradient>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">',
        '<feDropShadow dx="0" dy="8" stdDeviation="12" flood-opacity=".24"/>',
        "</filter></defs>",
        f'<rect width="{width}" height="{height}" rx="26" fill="url(#bg)"/>',
        '<text x="48" y="48" fill="#5eead4" font-family="system-ui,sans-serif" font-size="13" font-weight="750" letter-spacing="2">ROBUSTNESS UNDER BIOLOGICAL SHIFT</text>',
        '<text x="48" y="86" fill="#eef2ff" font-family="system-ui,sans-serif" font-size="28" font-weight="760">The generalization gap persists across seeds and assay conditions.</text>',
    ]

    n_seeds = int(gap_rows["seed"].nunique())
    n_conditions = len(conditions)
    maximum_overlap = int(transfer_rows["exact_variant_overlap"].max())
    stats = [
        (str(n_seeds), "repeated split seeds"),
        (f"{n_conditions}×{n_conditions}", "condition-transfer matrix"),
        (str(maximum_overlap), "exact variants shared"),
    ]
    for index, (value, label) in enumerate(stats):
        x = 48 + 205 * index
        parts.extend(
            [
                f'<text x="{x}" y="128" fill="#eef2ff" font-family="system-ui,sans-serif" font-size="24" font-weight="760">{escape(value)}</text>',
                f'<text x="{x}" y="150" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="12">{escape(label)}</text>',
            ]
        )

    parts.extend(
        [
            '<rect x="38" y="180" width="480" height="390" rx="18" fill="#111827" stroke="#263246" filter="url(#shadow)"/>',
            '<text x="66" y="218" fill="#eef2ff" font-family="system-ui,sans-serif" font-size="19" font-weight="720">Random → unseen-position penalty</text>',
            '<text x="66" y="241" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="12">Paired Spearman decrease for every split seed</text>',
        ]
    )
    x0, x1 = 92.0, 484.0
    maximum_gap = max(0.5, float(gap_rows["spearman_gap"].max()) * 1.1)
    for tick in np.linspace(0.0, maximum_gap, 6):
        x = x0 + (x1 - x0) * tick / maximum_gap
        parts.append(
            f'<line x1="{x:.1f}" y1="270" x2="{x:.1f}" y2="526" stroke="#253247"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="548" text-anchor="middle" fill="#64748b" font-family="system-ui,sans-serif" font-size="11">{tick:.2f}</text>'
        )

    target_y = np.linspace(330.0, 455.0, len(targets))
    for target, y in zip(targets, target_y, strict=True):
        values = gap_rows.loc[gap_rows["target"].eq(target), "spearman_gap"].to_numpy(
            dtype=float
        )
        p05, mean, p95 = np.quantile(values, 0.05), np.mean(values), np.quantile(values, 0.95)
        scale = lambda value: x0 + (x1 - x0) * value / maximum_gap
        parts.append(
            f'<text x="66" y="{y + 4:.1f}" fill="#cbd5e1" font-family="system-ui,sans-serif" font-size="12">{escape(str(target).replace("log_ec50_prot_", ""))}</text>'
        )
        parts.append(
            f'<line x1="{scale(p05):.1f}" y1="{y:.1f}" x2="{scale(p95):.1f}" y2="{y:.1f}" stroke="#a78bfa" stroke-width="5" stroke-linecap="round"/>'
        )
        for value in values:
            parts.append(
                f'<circle cx="{scale(value):.1f}" cy="{y:.1f}" r="4" fill="#c4b5fd" opacity=".65"/>'
            )
        parts.append(
            f'<circle cx="{scale(mean):.1f}" cy="{y:.1f}" r="8" fill="#fb7185" stroke="#eef2ff" stroke-width="2"><title>Mean gap {mean:.3f}; 5–95% seed range {p05:.3f}–{p95:.3f}</title></circle>'
        )
        parts.append(
            f'<text x="{min(scale(mean) + 14, 468):.1f}" y="{y + 4:.1f}" fill="#eef2ff" font-family="system-ui,sans-serif" font-size="12" font-weight="700">{mean:.3f}</text>'
        )

    parts.extend(
        [
            '<circle cx="76" cy="505" r="4" fill="#c4b5fd"/><text x="88" y="509" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="11">seed</text>',
            '<circle cx="150" cy="505" r="7" fill="#fb7185" stroke="#eef2ff" stroke-width="2"/><text x="163" y="509" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="11">mean</text>',
        ]
    )

    parts.extend(
        [
            '<rect x="538" y="180" width="604" height="390" rx="18" fill="#111827" stroke="#263246" filter="url(#shadow)"/>',
            '<text x="566" y="218" fill="#eef2ff" font-family="system-ui,sans-serif" font-size="19" font-weight="720">Cross-condition transfer at unseen positions</text>',
            '<text x="566" y="241" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="12">Additive model Spearman · source condition → target condition</text>',
        ]
    )
    cell = 14.0
    grid_x, grid_y = 720.0, 278.0
    for column_index, condition in enumerate(conditions):
        x = grid_x + column_index * cell + cell / 2
        parts.append(
            f'<text x="{x:.1f}" y="268" transform="rotate(-55 {x:.1f} 268)" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="7" text-anchor="start">{escape(_condition_label(condition))}</text>'
        )
    for row_index, source in enumerate(conditions):
        y = grid_y + row_index * cell
        parts.append(
            f'<text x="708" y="{y + 10:.1f}" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="7" text-anchor="end">{escape(_condition_label(source))}</text>'
        )
        for column_index, target in enumerate(conditions):
            value = float(matrix.loc[source, target])
            x = grid_x + column_index * cell
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="13" height="13" rx="2" fill="{_color(value, lower=transfer_min, upper=transfer_max)}"><title>{escape(_condition_label(source))} → {escape(_condition_label(target))}: {value:.3f}</title></rect>'
            )

    parts.extend(
        [
            '<text x="1055" y="296" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="10">Spearman</text>',
            '<rect x="1056" y="312" width="16" height="160" rx="8" fill="url(#transfer-scale)"/>',
            '<defs><linearGradient id="transfer-scale" x1="0" y1="1" x2="0" y2="0"><stop stop-color="#172554"/><stop offset="1" stop-color="#5eead4"/></linearGradient></defs>',
            f'<text x="1078" y="321" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="9">{transfer_max:.1f}</text>',
            f'<text x="1078" y="472" fill="#94a3b8" font-family="system-ui,sans-serif" font-size="9">{transfer_min:.1f}</text>',
            '<text x="590" y="592" fill="#64748b" font-family="system-ui,sans-serif" font-size="11">Aggregate experimental results only · raw measurements remain governed by the provider data-use agreement</text>',
            "</svg>",
        ]
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8")
    return output
