import csv
import random
from datetime import datetime, timedelta

random.seed(42)

start = datetime(2026, 1, 1)
rows = []

# 1. Normal transactions between 100 ordinary accounts
accounts = [f"ACC{i:03d}" for i in range(1, 101)]
for _ in range(400):
    sender, receiver = random.sample(accounts, 2)
    amount = round(random.uniform(50, 2000), 2)
    when = start + timedelta(
        days=random.randint(0, 89),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )
    rows.append((sender, receiver, amount, when))

# 2. Hidden fraud pattern A: a money loop (cycle)
ring = ["ACC901", "ACC902", "ACC903", "ACC904"]
ring_start = start + timedelta(days=45)
for i in range(4):
    rows.append((
        ring[i],
        ring[(i + 1) % 4],
        round(9500 - i * 100, 2),
        ring_start + timedelta(hours=i * 3),
    ))

# 3. Hidden fraud pattern B: a hub (many people pay into one account)
hub_start = start + timedelta(days=60)
for i in range(15):
    rows.append((
        random.choice(accounts),
        "ACC950",
        round(random.uniform(800, 1500), 2),
        hub_start + timedelta(hours=i * 5),
    ))
rows.append(("ACC950", "ACC999", 17000.00, hub_start + timedelta(days=4)))

# 4. Hidden fraud pattern C: a sudden burst (quiet account wakes up)
burst_start = start + timedelta(days=80)
for i in range(12):
    rows.append((
        "ACC960",
        random.choice(accounts),
        round(random.uniform(300, 900), 2),
        burst_start + timedelta(minutes=i * 5),
    ))

# Sort by time and save
rows.sort(key=lambda r: r[3])

with open("transactions.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source_account", "target_account", "amount", "timestamp"])
    for sender, receiver, amount, when in rows:
        writer.writerow([sender, receiver, amount, when.strftime("%Y-%m-%d %H:%M:%S")])

print(f"Saved {len(rows)} transactions to transactions.csv")
