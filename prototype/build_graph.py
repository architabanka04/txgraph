import pandas as pd
import networkx as nx

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

print("Accounts (dots):", G.number_of_nodes())
print("Transfers (arrows):", G.number_of_edges())
print()
print("Who ACC901 sends money to:", list(G.successors("ACC901")))
print("Who sends money to ACC950:", len(list(G.predecessors("ACC950"))), "accounts")
