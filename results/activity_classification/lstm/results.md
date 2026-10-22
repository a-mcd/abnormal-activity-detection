# Overall Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 44 | 0 | 0 | 0 | 0 |
| actual_t2 | 0 | 42 | 0 | 0 | 2 |
| actual_t3 | 0 | 0 | 43 | 1 | 0 |
| actual_t4 | 1 | 1 | 0 | 42 | 0 |
| actual_t5 | 0 | 8 | 1 | 0 | 35 |


## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.94 |


## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.01 | 0.98 | 0.99 |
| T2 | 0.95 | 0.05 | 0.82 | 0.88 |
| T3 | 0.98 | 0.01 | 0.98 | 0.98 |
| T4 | 0.95 | 0.01 | 0.98 | 0.97 |
| T5 | 0.80 | 0.01 | 0.95 | 0.86 |


# No Error Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 24 | 0 | 0 | 0 | 0 |
| actual_t2 | 0 | 22 | 0 | 0 | 2 |
| actual_t3 | 0 | 0 | 23 | 1 | 0 |
| actual_t4 | 1 | 0 | 0 | 23 | 0 |
| actual_t5 | 0 | 0 | 1 | 0 | 23 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.96 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.01 | 0.96 | 0.98 |
| T2 | 0.92 | 0.00 | 1.00 | 0.96 |
| T3 | 0.96 | 0.01 | 0.96 | 0.96 |
| T4 | 0.96 | 0.01 | 0.96 | 0.96 |
| T5 | 0.96 | 0.02 | 0.92 | 0.94 |


# Error Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 20 | 0 | 0 | 0 | 0 |
| actual_t2 | 0 | 20 | 0 | 0 | 0 |
| actual_t3 | 0 | 0 | 20 | 0 | 0 |
| actual_t4 | 0 | 1 | 0 | 19 | 0 |
| actual_t5 | 0 | 8 | 0 | 0 | 12 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.91 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.00 | 1.00 | 1.00 |
| T2 | 1.00 | 0.11 | 0.69 | 0.82 |
| T3 | 1.00 | 0.00 | 1.00 | 1.00 |
| T4 | 0.95 | 0.00 | 1.00 | 0.97 |
| T5 | 0.60 | 0.00 | 1.00 | 0.75 |