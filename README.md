https://www.youtube.com/watch?v=ieq8s0-3fBM

# Pose-Based Recognition and Form Quality Assessment

End-to-end pipeline:
1. Annotate reps/errors from videos.
2. Build a unified training dataset (`reps_dataset.npz`).
3. Train multitask model (`exercise`, `form`, `errors`).
4. Run inference on new video.

## Project Layout
- `src/annotate_reps.py`: rep annotation UI + error toggles.
- `src/make_dataset.py`: builds `data/processed/reps_dataset.npz` from `labels_reps.json`.
- `src/train.py`: trains multitask GRU from unified dataset.
- `src/infer.py`: predicts exercise/form/errors for one video.
- `src/infer_batch.py`: runs inference for all videos in a folder (e.g. `data/test`) and saves JSON/CSV report.
- `data/processed/error_taxonomy.json`: canonical error list per exercise.

## Canonical Exercise Names
- `squat`
- `bench_press`
- `latpulldown`
- `lateral_raises`
- `deadlift`

Aliases are normalized on input (e.g. `lat_pulldown -> latpulldown`).

## Stage 1 (stable demo on old classes)
Current recommended scope:
- `squat,bench_press,latpulldown`

### 1) Annotate reps and errors
```bash
python src/annotate_reps.py
```
Output:
- `data/processed/labels_reps.json`

### 2) Build dataset
Validate labels before building:
```bash
python src/validate_labels.py --labels data/processed/labels_reps.json --taxonomy data/processed/error_taxonomy.json
```

Then build dataset:
```bash
python src/make_dataset.py --include_exercises squat,bench_press,latpulldown
```
Output:
- `data/processed/reps_dataset.npz`

Dataset fields:
- `X` `(N,T,F)` where `F=132` (flattened normalized landmarks).
- `y_ex` `(N,)`
- `y_form` `(N,)`, derived as `1` if rep has zero active errors else `0`.
- `y_err` `(N,K)` multi-label error targets.
- `y_score` `(N,)`
- `ex_map`, `err_map`, `T`

### 3) Train model
```bash
python src/train.py --data data/processed/reps_dataset.npz --out data/processed/model.pt
```
Output:
- `data/processed/model.pt`

`train.py` prints:
- class distribution,
- form distribution,
- classification report for exercise and form,
- multi-label error F1 micro/macro at threshold `0.5`.

### 4) Inference
```bash
python src/infer.py --video path/to/video.mp4 --model data/processed/model.pt
```
Output:
- predicted exercise + probability
- predicted form + probability
- top error probabilities (if error head exists)

### 5) Batch inference for `data/test`
Put test videos into:
- `data/test` (supports nested folders)

Run:
```bash
python src/infer_batch.py --input_dir data/test --model data/processed/model.pt
```

Outputs:
- `data/processed/test_predictions.json`
- `data/processed/test_predictions.csv`

### 6) Evaluation + EDA
Run EDA and (when GT is available) confusion matrices/report:
```bash
python src/eval_eda.py --predictions data/processed/test_predictions.json --labels_reps data/processed/labels_reps.json
```

Optional explicit GT mapping:
```bash
python src/eval_eda.py --predictions data/processed/test_predictions.json --gt_map data/processed/test_gt_map.json
```

Interactive GT annotator for `data/test` videos:
```bash
python src/annotate_test_gt.py --input_dir data/test --out_json data/processed/eval_eda/gt_map_template.json
```

Outputs (default):
- `data/processed/eval_eda/eda_summary.json`
- `data/processed/eval_eda/eval_rows.csv`
- `data/processed/eval_eda/confusion_exercise.csv` (if GT matched)
- `data/processed/eval_eda/confusion_form.csv` (if GT form matched)

## Stage 2 (extend model to `deadlift` and `lateral_raises`)
1. Put videos into:
- `data/raw_videos/deadlift`
- `data/raw_videos/lateral_raises`
2. Annotate reps/errors in `annotate_reps.py`.
3. Rebuild dataset with extended scope:
```bash
python src/make_dataset.py --include_exercises all
```
4. Retrain:
```bash
python src/train.py --data data/processed/reps_dataset.npz --out data/processed/model.pt
```

## Validation Checklist
- `labels_reps.json` is non-empty.
- every rep interval satisfies `start <= bottom <= end`.
- no unknown exercises/errors relative to taxonomy.
- `train.py` runs without manual edits and saves `model.pt`.
- `infer.py` runs on at least 3 known videos without exceptions.

## Final Eval (Current Setup)
Use error-driven form decision:
```bash
python src/infer_batch.py --input_dir data/test --model data/processed/model_cv.pt --form_mode errors_threshold --error_threshold 0.25 --form_error_threshold 0.2 --error_threshold_map data/processed/error_threshold_map.json
python src/eval_eda.py --predictions data/processed/test_predictions.json --gt_map data/processed/eval_eda/gt_map_template.json --out_dir data/processed/eval_eda
```

Latest report is exported to:
- `reports/final_metrics.md`

## GitHub Upload Checklist
Before push:
1. Confirm generated videos/test data are ignored by `.gitignore`.
2. Keep code + lightweight configs only (`src`, `README.md`, `requirements.txt`, small JSON configs).
3. Do not push raw test videos (`data/test`) or large generated artifacts.
