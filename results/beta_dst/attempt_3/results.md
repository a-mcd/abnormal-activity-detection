# Overall Results

# Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 43 | 0 | 0 | 0 | 1 |
| actual_t2 | 0 | 33 | 0 | 2 | 9 |
| actual_t3 | 0 | 0 | 44 | 0 | 0 |
| actual_t4 | 0 | 0 | 0 | 43 | 1 |
| actual_t5 | 0 | 0 | 0 | 2 | 42 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.93 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 0.98 | 0.00 | 1.00 | 0.99 |
| T2 | 0.75 | 0.00 | 1.00 | 0.86 |
| T3 | 1.00 | 0.00 | 1.00 | 1.00 |
| T4 | 0.98 | 0.02 | 0.91 | 0.95 |
| T5 | 0.95 | 0.06 | 0.79 | 0.87 |

# No Error Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 23 | 0 | 0 | 0 | 1 |
| actual_t2 | 0 | 18 | 0 | 2 | 4 |
| actual_t3 | 0 | 0 | 24 | 0 | 0 |
| actual_t4 | 0 | 0 | 0 | 23 | 1 |
| actual_t5 | 0 | 0 | 0 | 0 | 24 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.93 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 0.96 | 0.00 | 1.00 | 0.98 |
| T2 | 0.75 | 0.00 | 1.00 | 0.86 |
| T3 | 1.00 | 0.00 | 1.00 | 1.00 |
| T4 | 0.96 | 0.02 | 0.92 | 0.94 |
| T5 | 1.00 | 0.06 | 0.80 | 0.89 |


# Error Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 20 | 0 | 0 | 0 | 0 |
| actual_t2 | 0 | 15 | 0 | 0 | 5 |
| actual_t3 | 0 | 0 | 20 | 0 | 0 |
| actual_t4 | 0 | 0 | 0 | 20 | 0 |
| actual_t5 | 0 | 0 | 0 | 2 | 18 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.93 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.00 | 1.00 | 1.00 |
| T2 | 0.75 | 0.00 | 1.00 | 0.86 |
| T3 | 1.00 | 0.00 | 1.00 | 1.00 |
| T4 | 1.00 | 0.02 | 0.91 | 0.95 |
| T5 | 0.90 | 0.06 | 0.78 | 0.84 |