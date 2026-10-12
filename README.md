# TxGraph

Transaction Network Anomaly Detector.

Draws bank transfers as a graph and flags suspicious accounts in red:
- Money loops: big amounts moving in a circle
- Hub accounts: many people pay in, then big money goes out
- Sudden bursts: many transfers in a short time

Built with Python, pandas, NetworkX and PyVis.

## How to run

python3 generate_data.py
python3 detect.py
open graph_flagged.html
