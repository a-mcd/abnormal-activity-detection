# Overall Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 44 | 0 | 0 | 0 | 0 |
| actual_t2 | 0 | 0 | 2 | 42 | 0 |
| actual_t3 | 0 | 0 | 44 | 0 | 0 |
| actual_t4 | 0 | 0 | 0 | 44 | 0 |
| actual_t5 | 0 | 0 | 6 | 38 | 0 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.60 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.00 | 1.00 | 1.00 |
| T2 | 0.00 | 0.00 | 0.00 | 0.00 |
| T3 | 1.00 | 0.05 | 0.85 | 0.92 |
| T4 | 1.00 | 0.45 | 0.35 | 0.52 |
| T5 | 0.00 | 0.00 | 0.00 | 0.00 |


# No Error Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 24 | 0 | 0 | 0 | 0 |
| actual_t2 | 0 | 0 | 1 | 23 | 0 |
| actual_t3 | 0 | 0 | 24 | 0 | 0 |
| actual_t4 | 0 | 0 | 0 | 24 | 0 |
| actual_t5 | 0 | 0 | 6 | 18 | 0 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.60 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.00 | 1.00 | 1.00 |
| T2 | 0.00 | 0.00 | 0.00 | 0.00 |
| T3 | 1.00 | 0.07 | 0.77 | 0.87 |
| T4 | 1.00 | 0.43 | 0.37 | 0.54 |
| T5 | 0.00 | 0.00 | 0.00 | 0.00 |

# Error Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 20 | 0 | 0 | 0 | 0 |
| actual_t2 | 0 | 0 | 1 | 19 | 0 |
| actual_t3 | 0 | 0 | 20 | 0 | 0 |
| actual_t4 | 0 | 0 | 0 | 20 | 0 |
| actual_t5 | 0 | 0 | 0 | 20 | 0 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.60 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.00 | 1.00 | 1.00 |
| T2 | 0.00 | 0.00 | 0.00 | 0.00 |
| T3 | 1.00 | 0.01 | 0.95 | 0.98 |
| T4 | 1.00 | 0.49 | 0.34 | 0.51 |
| T5 | 0.00 | 0.00 | 0.00 | 0.00 |