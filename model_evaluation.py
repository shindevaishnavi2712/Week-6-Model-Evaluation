from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TABLES = OUT / "tables"

def build_snapshots(transactions):
    rows = []
    for customer_id, group in transactions.groupby("customer_id"):
        for snapshot in pd.date_range("2024-07-01", "2026-02-01", freq="MS"):
            history = group[group.order_date < snapshot]
            if history.empty:
                continue
            future = group[(group.order_date >= snapshot) &
                           (group.order_date < snapshot + pd.Timedelta(days=60))]
            recent = history[history.order_date >= snapshot - pd.Timedelta(days=180)]
            rows.append({
                "customer_id": customer_id,
                "snapshot_date": snapshot,
                "recency_days": (snapshot - history.order_date.max()).days,
                "orders_last_180d": len(recent),
                "spend_last_180d": (recent.price * recent.quantity - recent.discount).sum(),
                "lifetime_orders": len(history),
                "lifetime_spend": (history.price * history.quantity - history.discount).sum(),
                "category_diversity": history.category.nunique(),
                "avg_rating": history.rating.mean(),
                "preferred_channel": history.channel.mode().iloc[0],
                "city": history.city.mode().iloc[0],
                "target_repeat_60d": int(len(future) > 0)
            })
    return pd.DataFrame(rows)

def main():
    DATA.mkdir(exist_ok=True); FIG.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True)
    transactions = pd.read_csv(DATA / "beauty_wellness_transactions.csv", parse_dates=["order_date"])
    snapshots = build_snapshots(transactions)
    snapshots.to_csv(DATA / "customer_snapshots.csv", index=False)

    target = "target_repeat_60d"
    features = ["recency_days", "orders_last_180d", "spend_last_180d", "lifetime_orders",
                "lifetime_spend", "category_diversity", "avg_rating", "preferred_channel", "city"]
    numeric = features[:-2]
    categorical = features[-2:]

    cutoff = pd.Timestamp("2025-10-01")
    train = snapshots[snapshots.snapshot_date < cutoff]
    test = snapshots[snapshots.snapshot_date >= cutoff]
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    model = Pipeline([("prep", prep),
                      ("classifier", RandomForestClassifier(n_estimators=300, max_depth=8,
                       min_samples_leaf=3, class_weight="balanced", random_state=42))])
    baseline = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train); baseline.fit(X_train, y_train)
    pred = model.predict(X_test); probability = model.predict_proba(X_test)[:, 1]
    base_pred = baseline.predict(X_test)

    def score(y, p, prob=None):
        result = {"Accuracy": accuracy_score(y,p),
                  "Precision": precision_score(y,p,zero_division=0),
                  "Recall": recall_score(y,p,zero_division=0),
                  "F1-score": f1_score(y,p,zero_division=0)}
        result["ROC-AUC"] = roc_auc_score(y,prob) if prob is not None and y.nunique() > 1 else np.nan
        return result

    metrics = pd.DataFrame({
        "Random Forest": score(y_test,pred,probability),
        "Majority baseline": score(y_test,base_pred)
    }).T
    metrics.to_csv(TABLES / "model_metrics.csv")
    pd.DataFrame(classification_report(y_test,pred,output_dict=True,zero_division=0)).T.to_csv(
        TABLES / "classification_report.csv")
    cm = confusion_matrix(y_test,pred)
    pd.DataFrame(cm,index=["Actual_No","Actual_Yes"],columns=["Predicted_No","Predicted_Yes"]).to_csv(
        TABLES / "confusion_matrix.csv")

    names = model.named_steps["prep"].get_feature_names_out()
    importance = model.named_steps["classifier"].feature_importances_
    fi = pd.DataFrame({"feature":names,"importance":importance}).sort_values("importance",ascending=False)
    fi.to_csv(TABLES / "feature_importance.csv",index=False)

    plt.figure(figsize=(7,5)); plt.imshow(cm); plt.colorbar()
    plt.title("Confusion Matrix — Repeat Purchase Within 60 Days")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks([0,1],["No","Yes"]); plt.yticks([0,1],["No","Yes"])
    for i in range(2):
        for j in range(2): plt.text(j,i,str(cm[i,j]),ha="center",va="center")
    plt.tight_layout(); plt.savefig(FIG/"confusion_matrix.png",dpi=160); plt.close()

    top=fi.head(10).sort_values("importance")
    plt.figure(figsize=(8,5)); plt.barh(top.feature,top.importance)
    plt.title("Top 10 Feature Importances"); plt.xlabel("Importance")
    plt.tight_layout(); plt.savefig(FIG/"feature_importance.png",dpi=160); plt.close()

    plt.figure(figsize=(7,5)); plt.bar(metrics.index,metrics["F1-score"])
    plt.title("F1-score: Model vs Baseline"); plt.ylabel("F1-score"); plt.ylim(0,1)
    plt.xticks(rotation=15,ha="right"); plt.tight_layout()
    plt.savefig(FIG/"model_vs_baseline_f1.png",dpi=160); plt.close()

    summary = f"""# Evaluation Results

- Snapshot rows: {len(snapshots)}
- Training rows: {len(train)}
- Test rows: {len(test)}
- Temporal cutoff: {cutoff.date()}
- Positive class rate in test set: {y_test.mean():.3f}

## Metrics

{metrics.to_string(float_format=lambda x: f"{x:.3f}")}

These metrics are generated from synthetic sample data and do not represent real market performance.
"""
    (OUT/"evaluation_summary.md").write_text(summary,encoding="utf-8")
    print(summary)

if __name__ == "__main__":
    main()
