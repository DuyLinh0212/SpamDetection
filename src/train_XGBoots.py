import os
import argparse
import subprocess

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier


LABEL_NAMES = ["Normal", "Spam"]
LABEL_VALUES = [0, 1]


def evaluate_split(model, X, y, split_name, output_dir):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "split": split_name,
        "accuracy": accuracy_score(y, pred),
        "auc": roc_auc_score(y, prob),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1_score": f1_score(y, pred, zero_division=0),
    }

    print(f"\nKet qua tren tap {split_name}:")
    print(f" -> Accuracy  : {metrics['accuracy']:.4f}")
    print(f" -> AUC       : {metrics['auc']:.4f}")
    print(f" -> Precision : {metrics['precision']:.4f}")
    print(f" -> Recall    : {metrics['recall']:.4f}")
    print(f" -> F1-Score  : {metrics['f1_score']:.4f}")
    print("\nBao cao chi tiet (Classification Report):")
    print(classification_report(y, pred, labels=LABEL_VALUES, target_names=LABEL_NAMES, zero_division=0))

    cm = confusion_matrix(y, pred, labels=LABEL_VALUES)
    cm_df = pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES)
    cm_df.to_csv(os.path.join(output_dir, f"{split_name.lower()}_confusion_matrix.csv"))

    return metrics, prob, pred, cm


def plot_evaluation_charts(results, output_path):
    fig, axes = plt.subplots(len(results), 2, figsize=(13, 5 * len(results)))
    if len(results) == 1:
        axes = [axes]

    for row_idx, result in enumerate(results):
        split_name = result["split"]
        y = result["y"]
        prob = result["prob"]
        pred = result["pred"]
        auc = result["metrics"]["auc"]

        fpr, tpr, _ = roc_curve(y, prob)
        axes[row_idx][0].plot(fpr, tpr, linewidth=2, label=f"XGBoost (AUC={auc:.3f})")
        axes[row_idx][0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random")
        axes[row_idx][0].set_title(f"ROC Curve - {split_name}")
        axes[row_idx][0].set_xlabel("False Positive Rate")
        axes[row_idx][0].set_ylabel("True Positive Rate")
        axes[row_idx][0].legend()
        axes[row_idx][0].grid(alpha=0.3)

        cm = confusion_matrix(y, pred, labels=LABEL_VALUES)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[row_idx][1],
            xticklabels=LABEL_NAMES,
            yticklabels=LABEL_NAMES,
        )
        axes[row_idx][1].set_xlabel("Du doan")
        axes[row_idx][1].set_ylabel("Thuc te")
        axes[row_idx][1].set_title(f"Confusion Matrix - {split_name}")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost Model on pre-processed data")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Thu muc chua train.csv, valid.csv, test.csv")
    parser.add_argument("--output-dir", type=str, default="models", help="Thu muc luu model va ket qua")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[1] Dang nap du lieu tu: {args.data_dir}")
    try:
        tr = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
        va = pd.read_csv(os.path.join(args.data_dir, "valid.csv"))
        te = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    except FileNotFoundError as e:
        print(f" -> LOI: Khong tim thay file. Chi tiet: {e}")
        print(" -> Hay chac chan ban da co train.csv, valid.csv, test.csv trong thu muc chi dinh.")
        return

    print(f" -> Tap Train: {len(tr)} dong | Valid: {len(va)} dong | Test: {len(te)} dong")

    X_tr, y_tr = tr.drop("label", axis=1), tr["label"]
    X_va, y_va = va.drop("label", axis=1), va["label"]
    X_te, y_te = te.drop("label", axis=1), te["label"]

    # Xu ly mat can bang nhan: spam thuong it hon normal.
    pos_weight = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())

    print("\n[2] Dang khoi tao va huan luyen XGBoost...")

    try:
        subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
        device = "cuda"
        print(" -> Tim thay GPU. Su dung device=cuda.")
    except Exception:
        device = "cpu"
        print(" -> Khong tim thay GPU. Su dung device=cpu.")

    model = XGBClassifier(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=pos_weight,
        tree_method="hist",
        device=device,
        early_stopping_rounds=50,
        random_state=args.random_state,
    )

    model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_va, y_va)], verbose=100)
    print(f" -> Vong lap tot nhat (Best iteration): {model.best_iteration}")

    print("\n[3] Danh gia mo hinh tren tap Validation va Test...")
    val_metrics, val_prob, val_pred, _ = evaluate_split(model, X_va, y_va, "valid", args.output_dir)
    test_metrics, test_prob, test_pred, _ = evaluate_split(model, X_te, y_te, "test", args.output_dir)

    metrics_df = pd.DataFrame([val_metrics, test_metrics])
    metrics_df.to_csv(os.path.join(args.output_dir, "evaluation_metrics.csv"), index=False)

    plot_evaluation_charts(
        [
            {"split": "Validation", "y": y_va, "prob": val_prob, "pred": val_pred, "metrics": val_metrics},
            {"split": "Test", "y": y_te, "prob": test_prob, "pred": test_pred, "metrics": test_metrics},
        ],
        os.path.join(args.output_dir, "evaluation_charts.png"),
    )

    # Ve va luu Feature Importance.
    imp = pd.DataFrame({"feature": X_tr.columns, "importance": model.feature_importances_}).sort_values(
        "importance",
        ascending=False,
    )
    plt.figure(figsize=(9, 7))
    sns.barplot(data=imp, x="importance", y="feature", palette="viridis")
    plt.title("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "feature_importance.png"))
    plt.close()

    print(f"\n[4] Dang luu mo hinh va ket qua vao thu muc '{args.output_dir}'...")
    joblib.dump(model, os.path.join(args.output_dir, "xgb_spam_model.pkl"))

    valid_pred_df = X_va.copy()
    valid_pred_df["label_true"] = y_va.values
    valid_pred_df["prob_spam"] = val_prob
    valid_pred_df["pred_label"] = val_pred
    valid_pred_df.to_csv(os.path.join(args.output_dir, "valid_predictions.csv"), index=False)

    test_pred_df = X_te.copy()
    test_pred_df["label_true"] = y_te.values
    test_pred_df["prob_spam"] = test_prob
    test_pred_df["pred_label"] = test_pred
    test_pred_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)

    print("XU LY HOAN TAT!")


if __name__ == "__main__":
    main()
