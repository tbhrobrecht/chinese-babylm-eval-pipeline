from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TASKS = [
    "zhoblimp",
    "hanzi_structure",
    "hanzi_pinyin",
    "word_fmri",
    "fmri",
    "afqmc",
    "ocnli",
    "tnews",
    "cluewsc2020",
]

TASK_LABELS = {
    "zhoblimp": "ZhoBLiMP",
    "hanzi_structure": "Hanzi structure",
    "hanzi_pinyin": "Hanzi pinyin",
    "word_fmri": "Word fMRI",
    "fmri": "fMRI",
    "afqmc": "AFQMC",
    "ocnli": "OCNLI",
    "tnews": "TNEWS",
    "cluewsc2020": "CLUEWSC2020",
}

TRACKS = {
    "NLU": ["zhoblimp", "afqmc", "ocnli", "tnews", "cluewsc2020"],
    "Hanzi": ["hanzi_structure", "hanzi_pinyin"],
    "Cog": ["word_fmri", "fmri"],
}

ALIASES = {
    "hanzi_struc": "hanzi_structure",
    "cluewsc20": "cluewsc2020",
}


@dataclass
class ModelRow:
    name: str
    source: str
    scores: dict[str, float | None]

    @property
    def mean(self) -> float | None:
        values = [self.scores[task] for task in TASKS if self.scores.get(task) is not None]
        if not values:
            return None
        return sum(values) / len(values)

    @property
    def coverage(self) -> str:
        present = sum(1 for task in TASKS if self.scores.get(task) is not None)
        return f"{present}/{len(TASKS)}"


def pct(value: Any) -> float | None:
    if value is None or value == "":
        return None
    number = float(value)
    return number * 100 if abs(number) <= 1.5 else number


def display_score(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def slug_to_name(path: Path) -> str:
    return path.stem.replace("_", " ")


def infer_result_name(path: Path) -> str:
    gpu_name = re.search(r"gpu\d+", path.stem, flags=re.IGNORECASE)
    if gpu_name:
        return gpu_name.group(0).lower()
    stem = path.stem
    for prefix in ("results_full_chinese_", "results_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
            break
    return stem.replace("_", " ") or slug_to_name(path)


def task_value(payload: dict[str, Any], task: str) -> float | None:
    item = payload.get(task)
    if item is None:
        return None
    if isinstance(item, dict):
        for metric in ("accuracy", "mean", "score", "value"):
            if metric in item:
                return pct(item[metric])
    return pct(item)


def load_baselines(path: Path) -> list[ModelRow]:
    rows: list[ModelRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            scores = {task: pct(raw.get(task)) for task in TASKS}
            rows.append(ModelRow(name=raw["model"], source="baseline", scores=scores))
    return rows


def load_model_json(path: Path, default_name: str | None = None) -> ModelRow:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")

    normalized = {ALIASES.get(key, key): value for key, value in payload.items()}
    name = str(normalized.get("model_name") or normalized.get("model") or default_name or slug_to_name(path))
    scores = {task: task_value(normalized, task) for task in TASKS}
    return ModelRow(name=name, source="candidate", scores=scores)


def load_candidate_models(
    models_dir: Path,
    current_results: Path | None,
    current_name: str,
    root_results_glob: str | None,
) -> list[ModelRow]:
    models = []
    loaded_paths = set()
    if current_results and current_results.exists():
        models.append(load_model_json(current_results, default_name=current_name))
        loaded_paths.add(current_results.resolve())
    if root_results_glob:
        root = Path(__file__).resolve().parent.parent
        for path in sorted(root.glob(root_results_glob)):
            if path.resolve() in loaded_paths:
                continue
            models.append(load_model_json(path, default_name=infer_result_name(path)))
            loaded_paths.add(path.resolve())
    for path in sorted(models_dir.glob("*.json")):
        if path.resolve() in loaded_paths:
            continue
        models.append(load_model_json(path))
    return models


def rank_for(rows: list[ModelRow], target: ModelRow, metric: str) -> int | None:
    target_value = target.mean if metric == "mean" else target.scores.get(metric)
    if target_value is None:
        return None
    values = []
    for row in rows:
        value = row.mean if metric == "mean" else row.scores.get(metric)
        if value is not None:
            values.append(value)
    return 1 + sum(value > target_value for value in values)


def percentile_against_baselines(baselines: list[ModelRow], value: float | None, task: str) -> float | None:
    if value is None:
        return None
    values = [row.scores[task] for row in baselines if row.scores.get(task) is not None]
    if not values:
        return None
    below_or_equal = sum(score <= value for score in values)
    return 100 * below_or_equal / len(values)


def write_summary_csv(path: Path, rows: list[ModelRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["source", "model", *TASKS, "mean", "coverage"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "source": row.source,
                    "model": row.name,
                    **{task: display_score(row.scores.get(task)) for task in TASKS},
                    "mean": display_score(row.mean),
                    "coverage": row.coverage,
                }
            )


def write_rankings_csv(path: Path, rows: list[ModelRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["task", "rank", "model", "source", "score"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for task in [*TASKS, "mean"]:
            ranked = sorted(
                (
                    row
                    for row in rows
                    if (row.mean if task == "mean" else row.scores.get(task)) is not None
                ),
                key=lambda row: row.mean if task == "mean" else row.scores[task],
                reverse=True,
            )
            for index, row in enumerate(ranked, 1):
                writer.writerow(
                    {
                        "task": task,
                        "rank": index,
                        "model": row.name,
                        "source": row.source,
                        "score": display_score(row.mean if task == "mean" else row.scores[task]),
                    }
                )


def score_bar(value: float | None, maximum: float = 100) -> str:
    if value is None:
        return '<span class="missing">Missing</span>'
    width = max(0, min(100, value / maximum * 100))
    return (
        '<div class="bar-track">'
        f'<span class="bar-fill" style="width:{width:.2f}%"></span>'
        f'<span class="bar-label">{value:.2f}</span>'
        "</div>"
    )


def task_distribution_svg(task: str, baselines: list[ModelRow], candidates: list[ModelRow]) -> str:
    baseline_values = [row.scores[task] for row in baselines if row.scores.get(task) is not None]
    if not baseline_values:
        return ""

    left_pad = 136
    width = 760
    axis_width = 560
    row_gap = 22
    height = 82 + row_gap * max(1, len(candidates))
    min_value = min(baseline_values)
    max_value = max(baseline_values)
    median_value = statistics.median(baseline_values)
    best = max(baselines, key=lambda row: row.scores.get(task) or -1)

    def x(value: float) -> float:
        return left_pad + (value / 100) * axis_width

    candidate_marks = []
    for index, row in enumerate(candidates):
        value = row.scores.get(task)
        if value is None:
            continue
        y = 46 + index * row_gap
        candidate_marks.append(
            f'<circle cx="{x(value):.1f}" cy="{y}" r="6" class="candidate-dot" />'
            f'<text x="{x(value) + 10:.1f}" y="{y + 4}" class="candidate-label">'
            f'{html.escape(row.name)} {value:.2f}</text>'
        )

    return f"""
<svg class="dist-chart" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(TASK_LABELS[task])} baseline distribution">
  <text x="0" y="50" class="task-name">{html.escape(TASK_LABELS[task])}</text>
  <line x1="{left_pad}" y1="50" x2="{left_pad + axis_width}" y2="50" class="axis" />
  <line x1="{x(min_value):.1f}" y1="50" x2="{x(max_value):.1f}" y2="50" class="range" />
  <circle cx="{x(min_value):.1f}" cy="50" r="4" class="range-end" />
  <circle cx="{x(max_value):.1f}" cy="50" r="4" class="range-end" />
  <line x1="{x(median_value):.1f}" y1="39" x2="{x(median_value):.1f}" y2="61" class="median" />
  {''.join(candidate_marks)}
  <text x="{left_pad}" y="{height - 12}" class="note">baseline min {min_value:.2f}, median {median_value:.2f}, best {max_value:.2f} ({html.escape(best.name)})</text>
</svg>"""


def render_track_table(track: str, tasks: list[str], baselines: list[ModelRow], candidates: list[ModelRow], all_rows: list[ModelRow]) -> str:
    header = "".join(f"<th>{html.escape(TASK_LABELS[task])}</th>" for task in tasks)
    body_rows = []
    for row in candidates:
        cells = []
        for task in tasks:
            value = row.scores.get(task)
            rank = rank_for(all_rows, row, task)
            percentile = percentile_against_baselines(baselines, value, task)
            style = "" if percentile is None else f' style="--strength:{percentile:.1f}%"'
            rank_text = "" if rank is None else f"<small>rank {rank}/{len(all_rows)}</small>"
            cells.append(
                f"<td{style}><strong>{display_score(value) or 'Missing'}</strong>{rank_text}</td>"
            )
        body_rows.append(f"<tr><th>{html.escape(row.name)}</th>{''.join(cells)}</tr>")
    return f"""
<section class="panel">
  <h2>{html.escape(track)} task ranks</h2>
  <div class="table-wrap">
    <table class="rank-table">
      <thead><tr><th>Model</th>{header}</tr></thead>
      <tbody>{''.join(body_rows)}</tbody>
    </table>
  </div>
</section>"""


def render_html(rows: list[ModelRow], baselines: list[ModelRow], candidates: list[ModelRow]) -> str:
    ranked_mean = sorted((row for row in rows if row.mean is not None), key=lambda row: row.mean or 0, reverse=True)
    baseline_means = [row.mean for row in baselines if row.mean is not None]
    best_baseline = max(baselines, key=lambda row: row.mean or -1)
    median_baseline = statistics.median(baseline_means)

    cards = []
    for row in candidates:
        rank = rank_for(rows, row, "mean")
        delta = None if row.mean is None else row.mean - median_baseline
        cards.append(
            f"""
<article class="metric-card candidate-card">
  <span>{html.escape(row.name)}</span>
  <strong>{display_score(row.mean) or 'Missing'}</strong>
  <small>mean rank {rank}/{len(ranked_mean)}; {display_score(delta)} vs baseline median</small>
</article>"""
        )

    cards.append(
        f"""
<article class="metric-card">
  <span>Best baseline</span>
  <strong>{best_baseline.mean:.2f}</strong>
  <small>{html.escape(best_baseline.name)}</small>
</article>"""
    )
    cards.append(
        f"""
<article class="metric-card">
  <span>Baseline median</span>
  <strong>{median_baseline:.2f}</strong>
  <small>{len(baselines)} README baselines</small>
</article>"""
    )

    leaderboard_rows = []
    for index, row in enumerate(ranked_mean, 1):
        class_name = "candidate-row" if row.source == "candidate" else ""
        leaderboard_rows.append(
            f"""
<tr class="{class_name}">
  <td>{index}</td>
  <th>{html.escape(row.name)}</th>
  <td>{html.escape(row.source)}</td>
  <td>{score_bar(row.mean)}</td>
  <td>{row.coverage}</td>
</tr>"""
        )

    distributions = "".join(task_distribution_svg(task, baselines, candidates) for task in TASKS)
    track_tables = "".join(
        render_track_table(track, tasks, baselines, candidates, rows)
        for track, tasks in TRACKS.items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Model result comparison</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #17202a;
      --muted: #5f6f7a;
      --line: #d8e0e6;
      --panel: #ffffff;
      --bg: #f5f7f8;
      --accent: #0f7c80;
      --accent-2: #c74f32;
      --fill: #3d8f73;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 20px 56px; }}
    h1 {{ margin: 0 0 6px; font-size: 32px; letter-spacing: 0; }}
    h2 {{ margin: 0 0 18px; font-size: 20px; letter-spacing: 0; }}
    p {{ margin: 0; color: var(--muted); }}
    .hero {{ margin-bottom: 24px; }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin: 22px 0; }}
    .metric-card, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(20, 37, 46, 0.04);
    }}
    .metric-card {{ padding: 16px; }}
    .metric-card span {{ display: block; color: var(--muted); font-size: 13px; }}
    .metric-card strong {{ display: block; margin: 4px 0; font-size: 30px; }}
    .metric-card small {{ color: var(--muted); }}
    .candidate-card {{ border-color: color-mix(in srgb, var(--accent) 42%, var(--line)); }}
    .panel {{ padding: 20px; margin-top: 16px; overflow: hidden; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 760px; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: middle; }}
    th {{ font-weight: 650; }}
    td {{ color: #26333b; }}
    tr:last-child td, tr:last-child th {{ border-bottom: 0; }}
    .candidate-row {{ background: #eef8f6; }}
    .bar-track {{ position: relative; height: 26px; min-width: 220px; background: #edf1f3; border-radius: 5px; overflow: hidden; }}
    .bar-fill {{ position: absolute; inset: 0 auto 0 0; background: var(--fill); }}
    .bar-label {{ position: relative; z-index: 1; display: inline-flex; height: 100%; align-items: center; padding-left: 8px; font-variant-numeric: tabular-nums; color: #102126; }}
    .missing {{ color: var(--muted); }}
    .dist-grid {{ display: grid; gap: 8px; }}
    .dist-chart {{ display: block; width: 100%; height: auto; border-bottom: 1px solid var(--line); }}
    .dist-chart:last-child {{ border-bottom: 0; }}
    .task-name {{ font-size: 15px; font-weight: 650; fill: var(--ink); }}
    .axis {{ stroke: #d4dde3; stroke-width: 8; stroke-linecap: round; }}
    .range {{ stroke: #8fa0aa; stroke-width: 8; stroke-linecap: round; }}
    .range-end {{ fill: #8fa0aa; }}
    .median {{ stroke: var(--ink); stroke-width: 3; }}
    .candidate-dot {{ fill: var(--accent-2); stroke: white; stroke-width: 2; }}
    .candidate-label, .note {{ font-size: 12px; fill: var(--muted); }}
    .rank-table td {{ background: linear-gradient(90deg, rgba(15, 124, 128, 0.16) var(--strength, 0%), transparent var(--strength, 0%)); }}
    .rank-table small {{ display: block; margin-top: 2px; color: var(--muted); }}
    .footer-note {{ margin-top: 14px; font-size: 13px; }}
    @media (max-width: 720px) {{
      main {{ padding: 24px 12px 40px; }}
      h1 {{ font-size: 26px; }}
      .panel {{ padding: 14px; }}
      .dist-chart {{ min-width: 760px; }}
      .dist-grid {{ overflow-x: auto; }}
    }}
  </style>
</head>
<body>
<main>
  <header class="hero">
    <h1>Model result comparison</h1>
    <p>Candidate models are compared against README baselines. All scores are percentages; higher is better.</p>
  </header>

  <section class="metrics">
    {''.join(cards)}
  </section>

  <section class="panel">
    <h2>Overall mean leaderboard</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Rank</th><th>Model</th><th>Source</th><th>Mean</th><th>Coverage</th></tr></thead>
        <tbody>{''.join(leaderboard_rows)}</tbody>
      </table>
    </div>
  </section>

  <section class="panel">
    <h2>Per-task position against baselines</h2>
    <div class="dist-grid">
      {distributions}
    </div>
    <p class="footer-note">Gray bars show the README baseline range. Black ticks show the baseline median. Orange dots show candidate model scores.</p>
  </section>

  {track_tables}
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare model result JSON files against README baselines.")
    root = Path(__file__).resolve().parent
    parser.add_argument("--baselines", type=Path, default=root / "baselines.csv")
    parser.add_argument("--current-results", type=Path, default=root.parent / "results.json")
    parser.add_argument("--current-name", default="current_model")
    parser.add_argument(
        "--root-results-glob",
        default="results*.json",
        help="Root-level result JSON files to include, for example results_full_chinese_gpu2.json.",
    )
    parser.add_argument("--models-dir", type=Path, default=root / "models")
    parser.add_argument("--out-dir", type=Path, default=root / "out")
    args = parser.parse_args()

    baselines = load_baselines(args.baselines)
    candidates = load_candidate_models(
        args.models_dir,
        args.current_results,
        args.current_name,
        args.root_results_glob,
    )
    rows = [*baselines, *candidates]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(args.out_dir / "summary.csv", rows)
    write_rankings_csv(args.out_dir / "rankings.csv", rows)
    (args.out_dir / "report.html").write_text(render_html(rows, baselines, candidates), encoding="utf-8")

    print(f"Wrote {args.out_dir / 'report.html'}")
    print(f"Wrote {args.out_dir / 'summary.csv'}")
    print(f"Wrote {args.out_dir / 'rankings.csv'}")


if __name__ == "__main__":
    main()
