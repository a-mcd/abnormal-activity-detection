# Overall Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 44 | 0 | 0 | 0 | 0 |
| actual_t2 | 0 | 38 | 1 | 4 | 1 |
| actual_t3 | 0 | 0 | 44 | 0 | 0 |
| actual_t4 | 0 | 0 | 1 | 42 | 1 |
| actual_t5 | 0 | 0 | 0 | 2 | 42 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.95 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.00 | 1.00 | 1.00 |
| T2 | 0.86 | 0.00 | 1.00 | 0.93 |
| T3 | 1.00 | 0.01 | 0.96 | 0.98 |
| T4 | 0.95 | 0.03 | 0.88 | 0.91 |
| T5 | 0.95 | 0.01 | 0.95 | 0.95 |

# No Error Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 24 | 0 | 0 | 0 | 0 |
| actual_t2 | 0 | 20 | 1 | 2 | 1 |
| actual_t3 | 0 | 0 | 24 | 0 | 0 |
| actual_t4 | 0 | 0 | 0 | 23 | 1 |
| actual_t5 | 0 | 0 | 0 | 0 | 24 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.96 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.00 | 1.00 | 1.00 |
| T2 | 0.83 | 0.00 | 1.00 | 0.91 |
| T3 | 1.00 | 0.01 | 0.96 | 0.98 |
| T4 | 0.96 | 0.02 | 0.92 | 0.94 |
| T5 | 1.00 | 0.02 | 0.92 | 0.96 |

# Error Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 20 | 0 | 0 | 0 | 0 |
| actual_t2 | 0 | 18 | 0 | 2 | 0 |
| actual_t3 | 0 | 0 | 20 | 0 | 0 |
| actual_t4 | 0 | 0 | 1 | 19 | 0 |
| actual_t5 | 0 | 0 | 0 | 2 | 18 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.95 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.00 | 1.00 | 1.00 |
| T2 | 0.90 | 0.00 | 1.00 | 0.95 |
| T3 | 1.00 | 0.01 | 0.95 | 0.98 |
| T4 | 0.95 | 0.05 | 0.83 | 0.88 |
| T5 | 0.90 | 0.00 | 1.00 | 0.95 |
