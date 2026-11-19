# Week 6 — Model Evaluation, Reporting & Business Impact

## Project
**Beauty & Wellness Market: Predicting Repeat Purchase Within 60 Days**

This Python project creates a customer-level prediction dataset from transaction history, trains a Random Forest classifier, compares it with a majority-class baseline, evaluates test performance, and produces visual summaries and business recommendations.

## Important dataset note
The included transaction data is **synthetic sample data created for this project**. It contains no real customer records. The evaluation results demonstrate a reproducible workflow only; they must not be described as actual market results. If your internship provides an approved dataset, replace the sample data and rerun the analysis.

## Prediction problem
At each monthly snapshot date, predict whether a customer will place at least one order in the following 60 days (`target_repeat_60d`). Features are calculated from transactions strictly before the snapshot date to reduce target leakage.

## Repository structure
```text
week6_model_evaluation_project/
├── data/
│   ├── beauty_wellness_transactions.csv
│   └── customer_snapshots.csv
├── src/
│   └── model_evaluation.py
├── outputs/
│   ├── evaluation_summary.md
│   ├── figures/
│   │   ├── confusion_matrix.png
│   │   ├── feature_importance.png
│   │   └── model_vs_baseline_f1.png
│   └── tables/
│       ├── model_metrics.csv
│       ├── classification_report.csv
│       ├── confusion_matrix.csv
│       └── feature_importance.csv
├── requirements.txt
└── README.md
```

## Model and evaluation
- Model: Random Forest Classifier
- Baseline: Most-frequent-class Dummy Classifier
- Split: temporal holdout (older snapshots for training, newer snapshots for testing)
- Metrics: Accuracy, Precision, Recall, F1-score and ROC-AUC
- Visuals: confusion matrix, feature importance and model-vs-baseline F1-score

## Business use
The model can help a beauty and wellness retailer prioritize customers for retention outreach. A high predicted probability can be used to test tailored replenishment reminders or loyalty offers. Before any real campaign, evaluate costs, customer consent, fairness, and incremental impact through a controlled A/B test. Model predictions are not proof that a campaign will cause a purchase.

## Run locally
1. Install Python 3.10+.
2. Install packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run:
   ```bash
   python src/model_evaluation.py
   ```

The script regenerates the customer snapshots, model, metrics, tables and charts under `outputs/`.

## Libraries
Python, pandas, NumPy, scikit-learn and Matplotlib.

## Author
Add your name here.
