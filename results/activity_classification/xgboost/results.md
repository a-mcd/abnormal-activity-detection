# Overall Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 43 | 1 | 0 | 0 | 0 |
| actual_t2 | 0 | 38 | 1 | 3 | 2 |
| actual_t3 | 0 | 0 | 42 | 0 | 2 |
| actual_t4 | 1 | 5 | 0 | 37 | 1 |
| actual_t5 | 0 | 8 | 0 | 2 | 34 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.88 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 0.98 | 0.01 | 0.98 | 0.98 |
| T2 | 0.86 | 0.08 | 0.73 | 0.79 |
| T3 | 0.95 | 0.01 | 0.98 | 0.97 |
| T4 | 0.84 | 0.03 | 0.88 | 0.86 |
| T5 | 0.77 | 0.03 | 0.87 | 0.82 |

# No Error Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 23 | 1 | 0 | 0 | 0 |
| actual_t2 | 0 | 19 | 1 | 2 | 2 |
| actual_t3 | 0 | 0 | 23 | 0 | 1 |
| actual_t4 | 1 | 0 | 0 | 22 | 1 |
| actual_t5 | 0 | 0 | 0 | 2 | 22 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.91 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 0.96 | 0.01 | 0.96 | 0.96 |
| T2 | 0.79 | 0.01 | 0.95 | 0.86 |
| T3 | 0.96 | 0.01 | 0.96 | 0.96 |
| T4 | 0.92 | 0.04 | 0.85 | 0.88 |
| T5 | 0.92 | 0.04 | 0.85 | 0.88 |


# Error Results

# Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 20 | 0 | 0 | 0 | 0 |
| actual_t2 | 0 | 19 | 0 | 1 | 0 |
| actual_t3 | 0 | 0 | 19 | 0 | 1 |
| actual_t4 | 0 | 5 | 0 | 15 | 0 |
| actual_t5 | 0 | 8 | 0 | 0 | 12 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.85 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.00 | 1.00 | 1.00 |
| T2 | 0.95 | 0.16 | 0.59 | 0.73 |
| T3 | 0.95 | 0.00 | 1.00 | 0.97 |
| T4 | 0.75 | 0.01 | 0.94 | 0.83 |
| T5 | 0.60 | 0.01 | 0.92 | 0.73 |