# Overall Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 44 | 0 | 0 | 0 | 0 |
| actual_t2 | 0 | 43 | 1 | 0 | 0 |
| actual_t3 | 0 | 0 | 44 | 0 | 0 |
| actual_t4 | 0 | 9 | 0 | 35 | 0 |
| actual_t5 | 0 | 17 | 0 | 1 | 26 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.87 |


## Per-Class Metrics
| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.00 | 1.00 | 1.00 |
| T2 | 0.98 | 0.15 | 0.62 | 0.76 |
| T3 | 1.00 | 0.01 | 0.98 | 0.99 |
| T4 | 0.80 | 0.01 | 0.97 | 0.88 |
| T5 | 0.59 | 0.01 | 1.00 | 0.74 |



# No Error Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual_t1 | 24 | 0 | 0 | 0 | 0 |
| actual_t2 | 0 | 23 | 1 | 0 | 0 |
| actual_t3 | 0 | 0 | 24 | 0 | 0 |
| actual_t4 | 0 | 1 | 0 | 23 | 0 |
| actual_t5 | 0 | 0 | 0 | 1 | 23 |


## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.98 |


## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.00 | 1.00 | 1.00 |
| T2 | 0.96 | 0.01 | 0.96 | 0.96 |
| T3 | 1.00 | 0.01 | 0.96 | 0.98 |
| T4 | 0.96 | 0.01 | 0.96 | 0.96 |
| T5 | 0.96 | 0.00 | 1.00 | 0.98 |


# Error Results

## Confusion Matrix

|        | pred_t1 | pred_t2 | pred_t3 | pred_t4 | pred_t5 |
|--------|--------:|--------:|--------:|--------:|--------:|
| actual t1 | 20 | 0 | 0 | 0 | 0 |
| actual t2 | 0 | 20 | 0 | 0 | 0 |
| actual t3 | 0 | 0 | 20 | 0 | 0 |
| actual t4 | 0 | 8 | 0 | 12 | 0 |
| actual t5 | 0 | 17 | 0 | 0 | 3 |

## Summary Metrics

| Metric | Value |
|--------|------:|
| Overall Accuracy | 0.75 |

## Per-Class Metrics

| Class | Recall | FPR | Precision | F1 |
|------|-------:|----:|----------:|---:|
| T1 | 1.00 | 0.00 | 1.00 | 1.00 |
| T2 | 1.00 | 0.31 | 0.44 | 0.62 |
| T3 | 1.00 | 0.00 | 1.00 | 1.00 |
| T4 | 0.60 | 0.00 | 1.00 | 0.75 |
| T5 | 0.15 | 0.00 | 1.00 | 0.26 |