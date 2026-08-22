"""Generate a dependency-free HTML report from benchmark results."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

SPLIT_LABELS = {
    "random_variant": "Random variant",
    "position_holdout": "Unseen position",
    "mutation_depth": "Higher mutation depth",
}
MODEL_LABELS = {
    "mean": "Mean",
    "biophysical_ridge": "Biophysical ridge",
    "additive_ridge": "Additive ridge",
}
SPLIT_COLORS = {
    "random_variant": "#5eead4",
    "position_holdout": "#fb7185",
    "mutation_depth": "#a78bfa",
}


def _metric(frame: pd.DataFrame, split: str, model: str, column: str) -> float:
    values = frame.loc[(frame["split"] == split) & (frame["model"] == model), column]
    if values.empty:
        raise ValueError(f"Missing result for split={split}, model={model}")
    return float(values.mean())


def _bar_chart(frame: pd.DataFrame) -> str:
    width, height = 780, 330
    left, top, plot_width, plot_height = 72, 30, 680, 230
    models = ["biophysical_ridge", "additive_ridge"]
    splits = ["random_variant", "position_holdout", "mutation_depth"]
    group_width = plot_width / len(models)
    bar_width = 74
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Spearman correlation by evaluation split">']
    for tick in range(0, 11, 2):
        value = tick / 10
        y = top + plot_height * (1 - value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="axis">{value:.1f}</text>')

    for group_index, model in enumerate(models):
        group_center = left + group_width * (group_index + 0.5)
        for split_index, split in enumerate(splits):
            value = max(0.0, _metric(frame, split, model, "spearman"))
            bar_height = value * plot_height
            x = group_center + (split_index - 1) * (bar_width + 10) - bar_width / 2
            y = top + plot_height - bar_height
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" '
                f'rx="7" fill="{SPLIT_COLORS[split]}"/>'
            )
            parts.append(
                f'<text x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" class="value">{value:.2f}</text>'
            )
        parts.append(
            f'<text x="{group_center:.1f}" y="{top + plot_height + 30}" text-anchor="middle" class="label">{MODEL_LABELS[model]}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _results_table(frame: pd.DataFrame) -> str:
    rows: list[str] = []
    for _, row in frame.sort_values(["target", "model", "split"]).iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(str(row['target']).replace('log_ec50_prot_', ''))}</td>"
            f"<td>{escape(MODEL_LABELS.get(row['model'], str(row['model'])))}</td>"
            f"<td>{escape(SPLIT_LABELS.get(row['split'], str(row['split'])))}</td>"
            f"<td>{row['spearman']:.3f}</td>"
            f"<td>{row['rmse']:.3f}</td>"
            f"<td>{row['observed_coverage']:.1%}</td>"
            f"<td>{int(row['test_rows']):,}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_report(results: pd.DataFrame, output: Path, *, filtered_rows: int) -> Path:
    required = {
        "split", "target", "model", "spearman", "rmse", "observed_coverage", "test_rows"
    }
    missing = required.difference(results.columns)
    if missing:
        raise ValueError(f"Benchmark file is missing columns: {', '.join(sorted(missing))}")

    random_score = _metric(results, "random_variant", "additive_ridge", "spearman")
    position_score = _metric(results, "position_holdout", "additive_ridge", "spearman")
    gap = random_score - position_score
    random_coverage = _metric(results, "random_variant", "additive_ridge", "observed_coverage")
    position_coverage = _metric(
        results, "position_holdout", "additive_ridge", "observed_coverage"
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VariantShift — TEV benchmark</title>
<style>
:root{{--ink:#eef2ff;--muted:#9ca3af;--panel:#111827;--line:#263246;--bg:#07111f}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 15% -10%,#14304d,transparent 38%),var(--bg);color:var(--ink);font:16px/1.55 ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1080px;margin:auto;padding:72px 28px 100px}} .eyebrow{{color:#5eead4;text-transform:uppercase;letter-spacing:.16em;font-size:12px;font-weight:800}}
h1{{font-size:clamp(42px,7vw,76px);line-height:1;margin:14px 0 22px;letter-spacing:-.05em;max-width:900px}} .lede{{font-size:21px;color:#cbd5e1;max-width:760px}}
.stats{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:42px 0}} .stat,.panel{{background:color-mix(in srgb,var(--panel) 88%,transparent);border:1px solid var(--line);border-radius:18px;padding:24px}}
.stat strong{{display:block;font-size:34px;letter-spacing:-.04em}} .stat span{{color:var(--muted)}} h2{{font-size:27px;margin:0 0 8px}} .finding{{font-size:22px;max-width:840px}}
.accent{{color:#fb7185;font-weight:800}} .grid{{stroke:#263246;stroke-width:1}} .axis,.label{{fill:#94a3b8;font-size:13px}} .value{{fill:#eef2ff;font-size:13px;font-weight:800}}
.legend{{display:flex;gap:22px;flex-wrap:wrap;color:#cbd5e1;margin:8px 0 20px}} .dot{{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:7px}}
table{{border-collapse:collapse;width:100%;font-size:14px}} th,td{{text-align:left;padding:12px;border-bottom:1px solid var(--line)}} th{{color:#94a3b8;font-size:11px;text-transform:uppercase;letter-spacing:.1em}}
.methods{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-top:14px}} .methods h3{{margin-top:0}} .methods p{{color:#aeb8c8}} footer{{margin-top:42px;color:#64748b;font-size:13px}}
@media(max-width:760px){{.stats,.methods{{grid-template-columns:1fr}}.table-wrap{{overflow:auto}}}}
</style>
</head>
<body><main>
<div class="eyebrow">Protein ML / Distribution shift</div>
<h1>Random splits hide where mutation models fail.</h1>
<p class="lede">VariantShift evaluates protein fitness predictors on unseen variants, unseen residue positions, and greater mutational depth using experimental TEV protease measurements.</p>
<section class="stats">
  <div class="stat"><strong>{filtered_rows:,}</strong><span>quality-filtered variants</span></div>
  <div class="stat"><strong>{random_score:.2f}</strong><span>random-split Spearman</span></div>
  <div class="stat"><strong>{position_score:.2f}</strong><span>unseen-position Spearman</span></div>
</section>
<section class="panel"><h2>Main result</h2>
<p class="finding">The additive model loses <span class="accent">{gap:.2f} Spearman</span> when every test residue position is absent from training. Its nominal 80% interval coverage falls from {random_coverage:.1%} to {position_coverage:.1%}, exposing uncertainty shift as well as ranking failure.</p>
</section>
<section class="panel" style="margin-top:14px"><h2>Generalization by evaluation regime</h2>
<div class="legend"><span><i class="dot" style="background:#5eead4"></i>Random variant</span><span><i class="dot" style="background:#fb7185"></i>Unseen position</span><span><i class="dot" style="background:#a78bfa"></i>Higher mutation depth</span></div>
{_bar_chart(results)}
</section>
<section style="margin-top:40px"><h2>Evaluation design</h2><div class="methods">
<div class="panel"><h3>Random variant</h3><p>Groups identical mutation strings, preventing exact variants from crossing the boundary while allowing familiar residue positions.</p></div>
<div class="panel"><h3>Unseen position</h3><p>Holds out complete residue positions. Mixed variants that bridge train and test residues are excluded.</p></div>
<div class="panel"><h3>Mutation depth</h3><p>Trains on single substitutions and evaluates on variants containing two to five substitutions.</p></div>
</div></section>
<section class="panel" style="margin-top:40px"><h2>All benchmark results</h2><div class="table-wrap"><table>
<thead><tr><th>Condition</th><th>Model</th><th>Split</th><th>Spearman</th><th>RMSE</th><th>80% coverage</th><th>Test n</th></tr></thead>
<tbody>{_results_table(results)}</tbody></table></div></section>
<footer>VariantShift · Align TEV GROQ-seq v1.1 · deterministic seed 42 · aggregate results only</footer>
</main></body></html>"""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    return output

