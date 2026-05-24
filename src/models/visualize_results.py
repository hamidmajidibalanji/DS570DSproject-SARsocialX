import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("outputs/model_comparison.csv")

metrics = ["Accuracy", "Precision", "Recall", "F1"]

for metric in metrics:

    plt.figure(figsize=(6, 4))

    plt.bar(df["Model"], df[metric])

    plt.title(f"{metric} Comparison")

    plt.ylabel(metric)

    plt.ylim(0, 1)

    plt.savefig(f"outputs/{metric.lower()}_comparison.png")

    plt.close()

print("Saved comparison plots.")