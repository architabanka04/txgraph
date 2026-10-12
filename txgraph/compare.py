"""Compare rules only, ML only, and rules + ML against the answer key."""
import sys

import pandas as pd

from txgraph.evaluate import score
from txgraph.graph import build_graph, load_transactions
from txgraph.risk import risk_table


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "data"
    df = load_transactions(f"{folder}/transactions.csv")
    actual = pd.read_csv(f"{folder}/labels.csv")["account"]
    table = risk_table(df, build_graph(df))
    table.to_csv(f"{folder}/risk_scores.csv")

    print(f"{'Method':<28}{'Flagged':>8}{'Correct':>9}{'Precision':>11}{'Recall':>8}{'F1':>6}")

    def line(name, picked):
        r = score(picked, actual)
        print(f"{name:<28}{r['found']:>8}{r['correct']:>9}{r['precision']:>11.0%}{r['recall']:>8.0%}{r['f1']:>6.2f}")

    line("Rules only", table.index[table["rule_hits"] > 0])
    for t in (50, 60, 70, 80):
        line(f"ML only (score >= {t})", table.index[table["anomaly_score"] >= t])
    for t in (50, 60, 70, 80):
        line(f"Rules + ML (risk >= {t})", table.index[table["risk_score"] >= t])

    top = table.head(10)[["risk_score", "anomaly_score", "rule_hits"]].copy()
    top["really_fraud"] = top.index.isin(actual)
    print("\nTop 10 riskiest accounts:")
    print(top.to_string())


if __name__ == "__main__":
    main()
