# Final Metrics Report

- Updated: `2026-02-15 19:14:07`
- Predictions rows: `52`
- GT labeled exercise: `51`
- GT labeled form: `51`

## Current Metrics

| Task | Accuracy | F1 Macro | F1 Weighted |
|---|---:|---:|---:|
| Exercise | 0.8039 | 0.8153 | 0.7970 |
| Form | 0.7255 | 0.7167 | 0.7196 |

## Delta vs Snapshot `20260215_190539`

| Metric | Previous | Current | Delta |
|---|---:|---:|---:|
| Exercise accuracy | 0.8431 | 0.8039 | -0.0392 |
| Exercise F1 macro | 0.8565 | 0.8153 | -0.0412 |
| Form accuracy | 0.6275 | 0.7255 | +0.0980 |
| Form F1 macro | 0.6269 | 0.7167 | +0.0898 |

## Hit/Miss by Class

| Task | Class | Hit | Miss | Total | Recall |
|---|---|---:|---:|---:|---:|
| exercise | bench_press | 8 | 1 | 9 | 88.9% |
| exercise | lateral_raises | 11 | 2 | 13 | 84.6% |
| exercise | latpulldown | 8 | 6 | 14 | 57.1% |
| exercise | romanian_deadlift | 5 | 0 | 5 | 100.0% |
| exercise | squat | 9 | 1 | 10 | 90.0% |
| form | incorrect | 14 | 10 | 24 | 58.3% |
| form | correct | 23 | 4 | 27 | 85.2% |

## Repro Commands

```bash
python src/infer_batch.py --input_dir data/test --model data/processed/model_cv.pt --form_mode errors_threshold --error_threshold 0.25 --form_error_threshold 0.2 --error_threshold_map data/processed/error_threshold_map.json
python src/eval_eda.py --predictions data/processed/test_predictions.json --gt_map data/processed/eval_eda/gt_map_template.json --out_dir data/processed/eval_eda
```