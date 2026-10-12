"""Load transactions and build the transaction graph."""
import networkx as nx
import pandas as pd

REQUIRED = ["source_account", "target_account", "amount", "timestamp"]


def load_transactions(path_or_file):
    df = pd.read_csv(path_or_file)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=REQUIRED)
    if "fraud_type" in df.columns:
        df["fraud_type"] = df["fraud_type"].fillna("")
    return df.sort_values("timestamp").reset_index(drop=True)


def build_graph(df):
    G = nx.DiGraph()
    grouped = df.groupby(["source_account", "target_account"]).agg(
        amount=("amount", "sum"),
        count=("amount", "size"),
        first_time=("timestamp", "min"),
        last_time=("timestamp", "max"),
    )
    for (s, r), row in grouped.iterrows():
        G.add_edge(
            s, r,
            amount=float(row["amount"]),
            count=int(row["count"]),
            first_time=row["first_time"],
            last_time=row["last_time"],
        )
    return G
