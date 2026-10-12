import pandas as pd
import networkx as nx
from pyvis.network import Network

df = pd.read_csv("transactions.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

G = nx.DiGraph()
for _, row in df.iterrows():
    s, r, a = row["source_account"], row["target_account"], row["amount"]
    if G.has_edge(s, r):
        G[s][r]["amount"] += a
        G[s][r]["count"] += 1
    else:
        G.add_edge(s, r, amount=a, count=1)

# reasons[account] = list of why it looks suspicious
reasons = {}

def flag(account, why):
    reasons.setdefault(account, []).append(why)

# Pattern A: money loops with big amounts (every step 5000 or more)
big = nx.DiGraph()
big.add_edges_from((s, r) for s, r, d in G.edges(data=True) if d["amount"] >= 5000)
for cycle in nx.simple_cycles(big, length_bound=5):
    if len(cycle) >= 3:
        for account in cycle:
            flag(account, "part of a money loop")

# Pattern B: hub accounts (many payers, then big money out)
for node in G.nodes():
    payers = G.in_degree(node)
    if payers >= 10 and G.out_degree(node) >= 1:
        flag(node, f"hub: {payers} accounts pay into it")

# Pattern C: sudden burst (8+ sends within 1 hour)
for account, group in df.groupby("source_account"):
    times = sorted(group["timestamp"])
    for i in range(len(times) - 7):
        if (times[i + 7] - times[i]).total_seconds() <= 3600:
            flag(account, "burst: 8+ transfers within 1 hour")
            break

print("Flagged accounts:")
for account, why in sorted(reasons.items()):
    print(" ", account, "->", "; ".join(sorted(set(why))))

# Draw the graph with risky accounts in red
net = Network(height="800px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")

for node in G.nodes():
    total = G.in_degree(node) + G.out_degree(node)
    risky = node in reasons
    net.add_node(
        node,
        label=node,
        size=8 + total * 2,
        color="#e03131" if risky else "#4a90d9",
        title=f"{node}: {total} connections" + (f" | RISK: {'; '.join(sorted(set(reasons[node])))}" if risky else ""),
    )

for s, r, data in G.edges(data=True):
    net.add_edge(s, r, value=data["amount"],
                 title=f"{s} -> {r}: {data['amount']:.2f} ({data['count']} transfers)")

net.write_html("graph_flagged.html")
print("Saved graph_flagged.html")
