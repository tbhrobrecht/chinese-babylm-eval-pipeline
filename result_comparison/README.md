# Result comparison dashboard

This folder compares one or more model result JSON files against the baseline table from the project README.

## Files

- `baselines.csv`: baseline scores copied from the README table, stored as percentages.
- Root-level `results*.json` files: loaded automatically. Files named like `results_full_chinese_gpu2.json` are shown as `gpu2`.
- `models/`: add extra JSON files here. The file name becomes the model name unless the JSON has a top-level `model_name` field.
- `compare_results.py`: regenerates the dashboard and CSV summaries.
- `out/`: generated reports.

## Generate the report

From the repository root:

```powershell
py result_comparison/compare_results.py
```

Then open:

```text
result_comparison/out/report.html
```

## Add another model

Root-level result files such as `results_full_chinese_gpu2.json` and `results_full_chinese_gpu3.json` are included automatically. You can also drop another exported results file into `result_comparison/models/`, for example:

```text
result_comparison/models/my_next_model.json
```

Run the script again. The report automatically adds the new model to the summary cards, rankings, per-task comparison, and CSV exports.

Expected JSON format is the leaderboard export format:

```json
{
  "zhoblimp": {"accuracy": 0.75},
  "hanzi_structure": {"accuracy": 0.60},
  "word_fmri": {"mean": 0.55}
}
```

Scores can be fractions such as `0.75` or percentages such as `75.0`; the script normalizes them to percentages.
