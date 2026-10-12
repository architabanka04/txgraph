import pandas as pd
import networkx as nx
from pyvis.network import Network

df = pd.read_csv("transactions.csv")

G = nx.DiGraph()

for _, row in df.iterrows():
    sender = row["source_account"]
    receiver = row["target_account"]
    amount = row["amount"]

    if G.has_edge(sender, receiver):
        G[sender][receiver]["amount"] += amount
        G[sender][receiver]["count"] += 1
    else:
        G.add_edge(sender, receiver, amount=amount, count=1)

net = Network(height="800px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")

for node in G.nodes():
    total_transfers = G.in_degree(node) + G.out_degree(node)
    net.add_node(
        node,
        label=node,
        size=8 + total_transfers * 2,
        color="#4a90d9",
        title=f"{node}: {total_transfers} connections",
    )

for sender, receiver, data in G.edges(data=True):
    net.add_edge(
        sender,
        receiver,
        value=data["amount"],
        title=f"{sender} -> {receiver}: {data['amount']:.2f} ({data['count']} transfers)",
    )

net.write_html("graph.html")
print("Saved graph.html")
