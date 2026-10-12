"""Isolation Forest anomaly score + rule hits = one risk score per account."""
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from txgraph.detectors import run_all
from txgraph.features import build_features

# Behaviour only. Size features (degree, totals) are left out so big shops are not flagged.
MODEL_COLUMNS = ["big_in", "big_out", "near_limit_senders", "fast_forwards", "max_sends_per_hour"]


def anomaly_scores(features, seed=42):
    X = features[MODEL_COLUMNS].copy()
    X["forward_speed"] = 1 / (1 + features["fastest_forward_hours"])
    X = np.log1p(X)
    model = IsolationForest(n_estimators=300, contamination="auto", random_state=seed)
    model.fit(X)
    raw = -model.score_samples(X)
    scaled = 100 * (raw - raw.min()) / (raw.max() - raw.min())
    return pd.Series(scaled, index=features.index).round(1)


def neighbour_scores(G, scores, min_amount=5000):
    """Money-weighted average risk of each account's big-money partners."""
    big = nx.Graph()
    for s, r, d in G.edges(data=True):
        if d["amount"] >= min_amount:
            w = big[s][r]["weight"] + d["amount"] if big.has_edge(s, r) else d["amount"]
            big.add_edge(s, r, weight=w)
    result = {}
    for a in scores.index:
        if a in big and big.degree(a) > 0:
            total = sum(big[a][n]["weight"] for n in big[a])
            result[a] = sum(scores[n] * big[a][n]["weight"] for n in big[a]) / total
        else:
            result[a] = 0.0
    return pd.Series(result).round(1)


def risk_table(df, G):
    rules = run_all(df, G)
    t = build_features(df, G)
    t["anomaly_score"] = anomaly_scores(t)
    t["rule_hits"] = [len(rules.get(a, [])) for a in t.index]
    t["rule_score"] = np.where(t["rule_hits"] > 0, np.minimum(100, 80 + 10 * (t["rule_hits"] - 1)), 0)
    t["neighbour_score"] = neighbour_scores(G, t["anomaly_score"])
    t["network_score"] = (0.7 * t["anomaly_score"] + 0.3 * t["neighbour_score"]).round(1)
    t["risk_score"] = t[["rule_score", "network_score"]].max(axis=1)
    t["reasons"] = ["; ".join(rules.get(a, [])) for a in t.index]
    return t.sort_values("risk_score", ascending=False)
