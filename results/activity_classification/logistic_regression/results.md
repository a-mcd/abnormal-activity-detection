# Overall Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 44 | 0 | 0 | 0 | 0 |
| actual_t2 | 0 | 41 | 1 | 1 | 1 |
| actual_t3 | 0 | 1 | 42 | 1 | 0 |
| actual_t4 | 3 | 7 | 0 | 33 | 1 |
| actual_t5 | 0 | 11 | 1 | 1 | 31 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.87 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.02 | 0.94 | 0.97 |
| T2 | 0.93 | 0.11 | 0.68 | 0.79 |
| T3 | 0.95 | 0.01 | 0.95 | 0.95 |
| T4 | 0.75 | 0.02 | 0.92 | 0.83 |
| T5 | 0.70 | 0.01 | 0.94 | 0.81 |


# No Error Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 24 | 0 | 0 | 0 | 0 |
| actual_t2 | 0 | 22 | 1 | 0 | 1 |
| actual_t3 | 0 | 1 | 22 | 1 | 0 |
| actual_t4 | 1 | 1 | 0 | 21 | 1 |
| actual_t5 | 0 | 1 | 1 | 0 | 22 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.93 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.01 | 0.96 | 0.98 |
| T2 | 0.92 | 0.03 | 0.88 | 0.90 |
| T3 | 0.92 | 0.02 | 0.92 | 0.92 |
| T4 | 0.88 | 0.01 | 0.95 | 0.91 |
| T5 | 0.92 | 0.02 | 0.92 | 0.92 |


# Error Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 20 | 0 | 0 | 0 | 0 |
| actual_t2 | 0 | 19 | 0 | 1 | 0 |
| actual_t3 | 0 | 0 | 20 | 0 | 0 |
| actual_t4 | 2 | 6 | 0 | 12 | 0 |
| actual_t5 | 0 | 10 | 0 | 1 | 9 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.80 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.03 | 0.91 | 0.95 |
| T2 | 0.95 | 0.20 | 0.54 | 0.69 |
| T3 | 1.00 | 0.00 | 1.00 | 1.00 |
| T4 | 0.60 | 0.03 | 0.86 | 0.71 |
| T5 | 0.45 | 0.00 | 1.00 | 0.62 |