"""Compare detected accounts with the answer key."""
import sys

import pandas as pd

from txgraph.detectors import run_all
from txgraph.graph import build_graph, load_transactions


def score(predicted, actual):
    predicted, actual = set(predicted), set(actual)
    tp = len(predicted & actual)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(actual) if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "found": len(predicted), "real_fraud": len(actual), "correct": tp,
        "precision": precision, "recall": recall, "f1": f1,
        "missed": sorted(actual - predicted),
        "false_alarms": sorted(predicted - actual),
    }


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"
    df = load_transactions(f"{folder}/transactions.csv")
    labels = pd.read_csv(f"{folder}/labels.csv")
    flagged = run_all(df, build_graph(df))
    r = score(flagged.keys(), labels["account"])
    print(f"Flagged: {r['found']}   Real fraud: {r['real_fraud']}   Correct: {r['correct']}")
    print(f"Precision: {r['precision']:.0%}   Recall: {r['recall']:.0%}   F1: {r['f1']:.2f}")
    print("Missed:", r["missed"])
    print("False alarms:", r["false_alarms"])


if __name__ == "__main__":
    main()
