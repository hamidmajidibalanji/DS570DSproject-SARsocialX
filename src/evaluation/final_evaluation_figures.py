import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

DATA_PATH = "data/processed/processed_disaster_tweets.csv"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_best_model():
    df = pd.read_csv(DATA_PATH)

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = joblib.load("models/best_svm_model.pkl")

    predictions = model.predict(X_test)

    # 1. Confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=["Non-disaster", "Disaster"]
    )

    plt.title("Confusion Matrix - Tuned SVM")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_tuned_svm.png", dpi=300)
    plt.close()

    # Save classification report
    report = classification_report(
        y_test,
        predictions,
        target_names=["Non-disaster", "Disaster"]
    )

    with open("results/classification_report_tuned_svm.txt", "w") as f:
        f.write(report)

    print(report)
    print("Saved confusion matrix and classification report.")


def create_model_comparison_chart():
    df = pd.read_csv("results/model_comparison.csv")

    df = df.sort_values("F1", ascending=False)

    # 2. Model comparison chart
    plt.figure(figsize=(10, 6))

    plt.bar(df["Model"], df["F1"])

    plt.ylabel("F1-score")
    plt.title("Model Comparison Based on F1-score")
    plt.xticks(rotation=25, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()

    plt.savefig("results/model_comparison_f1.png", dpi=300)
    plt.close()

    print("Saved model comparison chart.")


def create_final_results_table():
    df = pd.read_csv("results/model_comparison.csv")

    df = df.sort_values("F1", ascending=False)

    df["Accuracy"] = df["Accuracy"].round(4)
    df["Precision"] = df["Precision"].round(4)
    df["Recall"] = df["Recall"].round(4)
    df["F1"] = df["F1"].round(4)

    # 3. Final results table
    df.to_csv("results/final_results_table.csv", index=False)

    # Also save LaTeX table for Overleaf
    latex_table = df.to_latex(
        index=False,
        caption="Performance comparison of machine learning models.",
        label="tab:model_comparison"
    )

    with open("results/final_results_table.tex", "w") as f:
        f.write(latex_table)

    print("Saved final results table as CSV and LaTeX.")


if __name__ == "__main__":
    evaluate_best_model()
    create_model_comparison_chart()
    create_final_results_table()