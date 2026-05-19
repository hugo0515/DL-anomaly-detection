import numpy as np
import pandas as pd
import torch

from config import *
from train import train_autoencoder
from test import evaluate_autoencoder
from utils import set_seed


SEEDS = [42, 7, 21, 100, 2024]


def run_multiple_experiments():
    results = []

    for run_id, seed in enumerate(SEEDS, start=1):
        print("\n" + "=" * 60)
        print(f"Starting Run {run_id}/{len(SEEDS)} | Seed: {seed}")
        print("=" * 60)

        set_seed(seed)

        # Use different model/threshold file for each run
        run_model_path = f"autoencoder_{DATASET}_run_{run_id}.pth"
        run_threshold_path = f"threshold_{DATASET}_run_{run_id}.npy"

        # Train model
        train_autoencoder(
            seed=seed,
            model_path=run_model_path,
            threshold_path=run_threshold_path
        )

        # Evaluate final test performance
        metrics = evaluate_autoencoder(
            model_path=run_model_path,
            threshold_path=run_threshold_path
        )

        results.append({
            "Run": run_id,
            "Seed": seed,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-score": metrics["f1_score"],
        })

    results_df = pd.DataFrame(results)

    summary_df = results_df[
        ["Accuracy", "Precision", "Recall", "F1-score"]
    ].agg(["mean", "std"]).T

    summary_df["Mean ± Std"] = (
        summary_df["mean"].round(4).astype(str)
        + " ± "
        + summary_df["std"].round(4).astype(str)
    )

    print("\nIndividual Run Results:")
    print(results_df.to_string(index=False))

    print("\nFinal Mean ± Std Results:")
    print(summary_df[["Mean ± Std"]])

    results_df.to_csv(f"{DATASET}_5_run_results.csv", index=False)
    summary_df.to_csv(f"{DATASET}_mean_std_summary.csv")

    print("\nSaved:")
    print(f"- {DATASET}_5_run_results.csv")
    print(f"- {DATASET}_mean_std_summary.csv")


if __name__ == "__main__":
    run_multiple_experiments()