# Evaluation Results (Generated from included synthetic dataset)

- Snapshot rows: 8453
- Training rows: 5966
- Test rows: 2487
- Temporal cutoff: 2025-10-01
- Positive-class rate in test set: 0.199

## Test metrics

|                         |   Accuracy |   Precision |   Recall |   F1-score |   ROC-AUC |
|:------------------------|-----------:|------------:|---------:|-----------:|----------:|
| Model                   |      0.569 |       0.189 |    0.356 |      0.247 |     0.501 |
| Majority-class baseline |      0.801 |       0.000 |    0.000 |      0.000 |   nan     |

## Interpretation
The model estimates whether a customer will place at least one order in the 60 days after a monthly snapshot, using only information available before that snapshot. The temporal split uses snapshots before 2025-10-01 for training and snapshots on/after that date for testing. Results are specific to the synthetic sample dataset and are not evidence of real market performance.
