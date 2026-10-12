"""Turn each account into a row of numbers that describe its behaviour."""
import networkx as nx
import numpy as np
import pandas as pd


def _fast_forward(df, min_amount=5000, max_hours=48, min_keep=0.7):
    """How often money came in and a similar amount left soon after."""
    big = df[df["amount"] >= min_amount]
    outgoing = {a: g for a, g in big.groupby("source_account")}
    window = pd.Timedelta(hours=max_hours)
    counts, fastest = {}, {}
    for account, ins in big.groupby("target_account"):
        outs = outgoing.get(account)
        if outs is None:
            continue
        n, best = 0, np.inf
        for t_in, a_in in zip(ins["timestamp"], ins["amount"]):
            gaps = outs["timestamp"] - t_in
            ratio = outs["amount"] / a_in
            ok = (gaps > pd.Timedelta(0)) & (gaps <= window) & (ratio >= min_keep) & (ratio <= 1.0)
            if ok.any():
                n += 1
                best = min(best, gaps[ok].min().total_seconds() / 3600)
        counts[account], fastest[account] = n, best
    return counts, fastest


def _max_sends_per_hour(df):
    result = {}
    one_hour = np.timedelta64(1, "h")
    for account, g in df.groupby("source_account"):
        t = g["timestamp"].sort_values().values
        j, best = 0, 0
        for i in range(len(t)):
            while t[i] - t[j] > one_hour:
                j += 1
            best = max(best, i - j + 1)
        result[account] = best
    return result


def build_features(df, G, limit=50000):
    pagerank = nx.pagerank(G, weight="amount")
    ff_count, ff_fast = _fast_forward(df)
    per_hour = _max_sends_per_hour(df)
    near = df[(df["amount"] >= limit * 0.7) & (df["amount"] < limit)]
    near_senders = near.groupby("target_account")["source_account"].nunique()
    big = df[df["amount"] >= 10000]
    big_in = big.groupby("target_account").size()
    big_out = big.groupby("source_account").size()
    total_in = df.groupby("target_account")["amount"].sum()
    total_out = df.groupby("source_account")["amount"].sum()

    rows = []
    for a in sorted(G.nodes()):
        tin, tout = total_in.get(a, 0), total_out.get(a, 0)
        rows.append({
            "account": a,
            "in_degree": G.in_degree(a),
            "out_degree": G.out_degree(a),
            "total_in": round(tin, 2),
            "total_out": round(tout, 2),
            "pass_through": round(min(tin, tout) / max(tin, tout), 3) if max(tin, tout) else 0,
            "big_in": int(big_in.get(a, 0)),
            "big_out": int(big_out.get(a, 0)),
            "near_limit_senders": int(near_senders.get(a, 0)),
            "fast_forwards": ff_count.get(a, 0),
            "fastest_forward_hours": round(min(ff_fast.get(a, 999), 999), 2),
            "max_sends_per_hour": per_hour.get(a, 0),
            "pagerank": pagerank[a],
        })
    return pd.DataFrame(rows).set_index("account")
