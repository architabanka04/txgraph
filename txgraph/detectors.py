"""Rule-based fraud detectors. Each returns {account: [reason, ...]}."""
from collections import defaultdict

import networkx as nx
import pandas as pd


def _merge(*results):
    merged = defaultdict(list)
    for result in results:
        for account, reasons in result.items():
            for reason in reasons:
                if reason not in merged[account]:
                    merged[account].append(reason)
    return dict(merged)


def detect_loops(G, min_amount=10000, max_days=3, max_len=6, min_ratio=0.8):
    """Big amounts going round in a circle, fast, with similar sizes."""
    big = nx.DiGraph()
    for s, r, d in G.edges(data=True):
        if d["amount"] >= min_amount:
            big.add_edge(s, r, **d)
    found = defaultdict(list)
    for cycle in nx.simple_cycles(big, length_bound=max_len):
        if len(cycle) < 3:
            continue
        edges = [big[cycle[i]][cycle[(i + 1) % len(cycle)]] for i in range(len(cycle))]
        amounts = [e["amount"] for e in edges]
        span = max(e["last_time"] for e in edges) - min(e["first_time"] for e in edges)
        if span <= pd.Timedelta(days=max_days) and min(amounts) / max(amounts) >= min_ratio:
            for account in cycle:
                found[account].append(f"loop: {len(cycle)} accounts, about {min(amounts):,.0f} per step")
    return dict(found)


def detect_smurfing(df, limit=50000, near=0.9, min_senders=5, window_days=7):
    """Many different people each sending just under the reporting limit to one account."""
    near_df = df[(df["amount"] >= limit * near) & (df["amount"] < limit)]
    window = pd.Timedelta(days=window_days)
    found = {}
    for target, group in near_df.groupby("target_account"):
        group = group.sort_values("timestamp")
        times = group["timestamp"].tolist()
        senders = group["source_account"].tolist()
        amounts = group["amount"].tolist()
        for i in range(len(times)):
            j = i
            while j < len(times) and times[j] - times[i] <= window:
                j += 1
            unique = set(senders[i:j])
            if len(unique) >= min_senders:
                found[target] = [f"smurfing: {len(unique)} senders just under {limit:,} (total {sum(amounts[i:j]):,.0f})"]
                break
    return found


def detect_layering(df, min_amount=10000, max_minutes=180, min_keep=0.85, min_hops=3):
    """Money passed quickly down a chain of accounts, a little smaller each time."""
    big = df[df["amount"] >= min_amount].reset_index(drop=True)
    by_source = defaultdict(list)
    for i, src in enumerate(big["source_account"]):
        by_source[src].append(i)

    window = pd.Timedelta(minutes=max_minutes)
    nxt = defaultdict(list)
    has_parent = set()
    for i in range(len(big)):
        t_i, a_i = big.at[i, "timestamp"], big.at[i, "amount"]
        for j in by_source.get(big.at[i, "target_account"], []):
            gap = big.at[j, "timestamp"] - t_i
            ratio = big.at[j, "amount"] / a_i
            if pd.Timedelta(0) < gap <= window and min_keep <= ratio <= 1.0:
                nxt[i].append(j)
                has_parent.add(j)

    depth, best = {}, {}
    for i in reversed(range(len(big))):
        depth[i], best[i] = 1, None
        for j in nxt[i]:
            if depth[j] + 1 > depth[i]:
                depth[i], best[i] = depth[j] + 1, j

    found = defaultdict(list)
    for i in range(len(big)):
        if i in has_parent or depth[i] < min_hops:
            continue
        chain, k = [], i
        while k is not None:
            chain.append(k)
            k = best[k]
        minutes = (big.at[chain[-1], "timestamp"] - big.at[chain[0], "timestamp"]).total_seconds() / 60
        for k in chain[:-1]:
            found[big.at[k, "target_account"]].append(
                f"layering: money passed along {len(chain) + 1} accounts in {minutes:.0f} min")
    return dict(found)


def detect_bursts(df, min_count=10, window_minutes=60):
    """An account suddenly sending many transfers in a short time."""
    window = pd.Timedelta(minutes=window_minutes)
    found = {}
    for account, group in df.groupby("source_account"):
        times = group["timestamp"].sort_values().tolist()
        for i in range(len(times) - min_count + 1):
            if times[i + min_count - 1] - times[i] <= window:
                found[account] = [f"burst: {min_count}+ transfers sent within {window_minutes} min"]
                break
    return found


def run_all(df, G):
    return _merge(
        detect_loops(G),
        detect_smurfing(df),
        detect_layering(df),
        detect_bursts(df),
    )
