"""TxGraph dashboard: run with  streamlit run app.py"""
import io
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from pyvis.network import Network

from txgraph.evaluate import score
from txgraph.generate import Generator
from txgraph.graph import build_graph, load_transactions
from txgraph.risk import risk_table

st.set_page_config(page_title="TxGraph", layout="wide")

SAMPLES = {"Sample data (normal)": ("data", False), "Sample data (hard mode)": ("data_hard", True)}


def ensure_sample(folder, hard):
    path = Path(folder)
    if not (path / "transactions.csv").exists():
        df, labels = Generator(hard=hard).build()
        path.mkdir(exist_ok=True)
        df.to_csv(path / "transactions.csv", index=False)
        labels.to_csv(path / "labels.csv", index=False)
    return path


@st.cache_resource(show_spinner="Building graph and scoring accounts...")
def analyse(key, file_bytes=None):
    if file_bytes is None:
        path = ensure_sample(*SAMPLES[key])
        df = load_transactions(path / "transactions.csv")
        labels = pd.read_csv(path / "labels.csv")["account"].tolist()
    else:
        df = load_transactions(io.BytesIO(file_bytes))
        labels = None
    G = build_graph(df)
    return df, G, risk_table(df, G), labels


def draw(H, table, cutoff, height=600):
    net = Network(height=f"{height}px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
    for n in H.nodes():
        risk = float(table.at[n, "risk_score"]) if n in table.index else 0.0
        color = "#e03131" if risk >= cutoff else "#f59f00" if risk >= cutoff - 15 else "#4a90d9"
        net.add_node(n, label=n, size=10 + risk / 5, color=color, title=f"{n} | risk {risk:.0f}")
    for s, r, d in H.edges(data=True):
        net.add_edge(s, r, value=float(d["amount"]),
                     title=f"{s} -> {r}: {d['amount']:,.0f} ({d['count']} transfers)")
    components.html(net.generate_html(), height=height + 20)


st.title("TxGraph")
st.caption("Transaction Network Anomaly Detector: rules + graph features + Isolation Forest")

st.sidebar.header("Data")
choice = st.sidebar.radio("Source", list(SAMPLES) + ["Upload my CSV"])
if choice == "Upload my CSV":
    up = st.sidebar.file_uploader("CSV with source_account, target_account, amount, timestamp", type="csv")
    if up is None:
        st.info("Upload a CSV to begin, or pick a sample in the sidebar.")
        st.stop()
    try:
        df, G, table, labels = analyse(up.name, up.getvalue())
    except ValueError as e:
        st.error(str(e))
        st.stop()
else:
    df, G, table, labels = analyse(choice)

st.sidebar.header("Detector")
if "cutoff" not in st.session_state:
    st.session_state["cutoff"] = 50
cutoff = st.sidebar.slider("Risk cut-off", 0, 100, key="cutoff", help="Accounts at or above this score are flagged")
flagged = table[table["risk_score"] >= cutoff]

cols = st.columns(5 if labels else 3)
cols[0].metric("Accounts", f"{G.number_of_nodes():,}")
cols[1].metric("Transactions", f"{len(df):,}")
cols[2].metric("Flagged", len(flagged))
if labels:
    r = score(flagged.index, labels)
    cols[3].metric("Precision", f"{r['precision']:.0%}")
    cols[4].metric("Recall", f"{r['recall']:.0%}")

tab_table, tab_graph, tab_account, tab_scores = st.tabs(
    ["Risk ranking", "Fraud network", "Inspect an account", "Score explorer"])

with tab_table:
    show = flagged[["risk_score", "anomaly_score", "neighbour_score", "rule_hits", "reasons",
                    "in_degree", "out_degree", "total_in", "total_out"]].copy()
    if labels:
        show.insert(0, "really_fraud", show.index.isin(labels))
    st.dataframe(show, height=500)
    st.download_button("Download flagged accounts (CSV)", show.to_csv().encode(),
                       "flagged_accounts.csv", "text/csv")

with tab_graph:
    st.write("Flagged accounts (red) and their big-money partners (5,000+). Orange = close to the cut-off.")
    if flagged.empty:
        st.success("No accounts above the cut-off.")
    else:
        keep = set(flagged.index)
        for a in flagged.index:
            for n in set(G.successors(a)) | set(G.predecessors(a)):
                amount = G[a][n]["amount"] if G.has_edge(a, n) else 0
                amount += G[n][a]["amount"] if G.has_edge(n, a) else 0
                if amount >= 5000:
                    keep.add(n)
        H = G.subgraph(keep)
        st.caption(f"{H.number_of_nodes()} accounts, {H.number_of_edges()} links")
        draw(H, table, cutoff, height=650)

with tab_account:
    options = list(flagged.index) + [a for a in table.index if a not in flagged.index]
    acc = st.selectbox("Account (flagged ones are listed first)", options)
    row = table.loc[acc]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk score", f"{row['risk_score']:.0f}")
    c2.metric("ML score", f"{row['anomaly_score']:.0f}")
    c3.metric("Partner score", f"{row['neighbour_score']:.0f}")
    c4.metric("Rule hits", int(row["rule_hits"]))
    if row["reasons"]:
        st.warning(row["reasons"])
    if labels:
        st.write("Answer key:", "**fraud**" if acc in labels else "normal")
    left, right = st.columns([3, 2])
    with left:
        draw(nx.ego_graph(G, acc, radius=1, undirected=True), table, cutoff, height=450)
    with right:
        st.write("Behaviour numbers")
        st.dataframe(row.drop(["reasons"]).astype(float).round(3).to_frame("value"), height=450)
    tx = df[(df["source_account"] == acc) | (df["target_account"] == acc)]
    st.write(f"Transactions ({len(tx)})")
    st.dataframe(tx.drop(columns=["fraud_type"], errors="ignore"), height=300)

with tab_scores:
    st.write("How risk scores are spread out. Move the cut-off slider and watch the dashed line move.")
    fig = go.Figure()
    if labels:
        is_fraud = table.index.isin(labels)
        fig.add_histogram(x=table.loc[~is_fraud, "risk_score"], name="Normal", nbinsx=50,
                          marker_color="#4a90d9", opacity=0.75)
        fig.add_histogram(x=table.loc[is_fraud, "risk_score"], name="Fraud", nbinsx=50,
                          marker_color="#e03131", opacity=0.75)
    else:
        fig.add_histogram(x=table["risk_score"], name="Accounts", nbinsx=50, marker_color="#4a90d9")
    fig.add_vline(x=cutoff, line_dash="dash", annotation_text=f"cut-off {cutoff}")
    fig.update_layout(barmode="overlay", xaxis_title="Risk score",
                      yaxis_title="Number of accounts", height=380)
    st.plotly_chart(fig, key="score_histogram")

    if labels:
        rows = []
        for t in range(0, 101, 2):
            r = score(table.index[table["risk_score"] >= t], labels)
            rows.append({"cutoff": t, "Precision": r["precision"], "Recall": r["recall"], "F1": r["f1"]})
        curve = pd.DataFrame(rows)
        best = int(curve.loc[curve["F1"].idxmax(), "cutoff"])
        best_f1 = curve["F1"].max()

        fig2 = go.Figure()
        for col, color in [("Precision", "#2f9e44"), ("Recall", "#e03131"), ("F1", "#7048e8")]:
            fig2.add_scatter(x=curve["cutoff"], y=curve[col], name=col, line_color=color)
        fig2.add_vline(x=cutoff, line_dash="dash", annotation_text="you are here")
        fig2.update_layout(xaxis_title="Cut-off", yaxis_title="Score", yaxis_tickformat=".0%",
                           height=380, hovermode="x unified")
        st.plotly_chart(fig2, key="cutoff_curve")

        st.info(f"Best balance: cut-off {best} (F1 = {best_f1:.2f}).")
        st.button("Jump to the best cut-off",
                  on_click=lambda: st.session_state.update(cutoff=best))
