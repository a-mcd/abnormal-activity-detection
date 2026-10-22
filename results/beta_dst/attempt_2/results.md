# Overall Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 43 | 0 | 0 | 0 | 1 |
| actual_t2 | 0 | 1 | 0 | 10 | 33 |
| actual_t3 | 0 | 0 | 44 | 0 | 0 |
| actual_t4 | 0 | 0 | 0 | 39 | 5 |
| actual_t5 | 0 | 0 | 0 | 2 | 42 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.77 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 0.98 | 0.00 | 1.00 | 0.99 |
| T2 | 0.02 | 0.00 | 1.00 | 0.40 |
| T3 | 1.00 | 0.00 | 1.00 | 1.00 |
| T4 | 0.89 | 0.07 | 0.76 | 0.82 |
| T5 | 0.95 | 0.22 | 0.52 | 0.67 |


# No Error Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 23 | 0 | 0 | 0 | 1 |
| actual_t2 | 0 | 1 | 0 | 4 | 19 |
| actual_t3 | 0 | 0 | 24 | 0 | 0 |
| actual_t4 | 0 | 0 | 0 | 20 | 4 |
| actual_t5 | 0 | 0 | 0 | 0 | 24 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.77 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 0.96 | 0.00 | 1.00 | 0.98 |
| T2 | 0.04 | 0.00 | 1.00 | 0.08 |
| T3 | 1.00 | 0.00 | 1.00 | 1.00 |
| T4 | 0.83 | 0.04 | 0.83 | 0.83 |
| T5 | 1.00 | 0.25 | 0.50 | 0.67 |


# Error Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 20 | 0 | 0 | 0 | 0 |
| actual_t2 | 0 | 0 | 0 | 6 | 14 |
| actual_t3 | 0 | 0 | 20 | 0 | 0 |
| actual_t4 | 0 | 0 | 0 | 19 | 1 |
| actual_t5 | 0 | 0 | 0 | 2 | 18 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.77 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.00 | 1.00 | 1.00 |
| T2 | 0.00 | 0.00 | 0.00 | 0.00 |
| T3 | 1.00 | 0.00 | 1.00 | 1.00 |
| T4 | 0.95 | 0.10 | 0.70 | 0.81 |
| T5 | 0.9o | 0.19 | 0.55 | 0.68 |