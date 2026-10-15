# class_results

Output directory for the RoBERTa multi-label call classifier in
[`../class_train/`](../class_train/README.md). It is empty until you run the
training; the files that appear here are:

| File | Written by | Content |
|------|-----------|---------|
| `train.log`, `training_args.json` | `train.py` | run log and exact hyperparameters |
| `epoch_metrics.csv` / `.json` | `train.py` | per-epoch train loss, validation BCE loss, micro/macro F1 @0.5, subset accuracy, Hamming loss, primary-label accuracy, lr, grad norm |
| `summary.json` | `train.py` | best epoch, best validation loss, final numbers, total time |
| `test_metrics.json` | `evaluate.py` | test metrics at 0.5 and tuned thresholds, per-label P/R/F1, primary-label confusion matrix, breakdown by noise level |
| `thresholds.json` | `evaluate.py` | per-label decision thresholds tuned on validation |
| `test_errors.jsonl` | `evaluate.py` | mislabelled test calls with probabilities |
| `loss_curve.png` | `auth_train/plot_losses.py` | train / validation loss per epoch |

Run everything with `./class_train/run.sh`.
