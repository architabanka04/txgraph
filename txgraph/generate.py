"""Make a realistic fake transaction dataset with hidden, labelled fraud."""
import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

START = datetime(2026, 1, 1)
DAYS = 90


class Generator:
    def __init__(self, n_accounts=500, n_normal=6000, seed=42, hard=False):
        self.hard = hard
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        accounts = [f"ACC{i:04d}" for i in range(1, n_accounts + 1)]
        self.merchants = accounts[:20]
        self.people = accounts[20:]
        self.n_normal = n_normal
        self.rows = []
        self.labels = {}

    def add(self, src, dst, amount, when, fraud_type=""):
        self.rows.append({
            "source_account": src,
            "target_account": dst,
            "amount": round(float(amount), 2),
            "timestamp": when,
            "fraud_type": fraud_type,
        })

    def label(self, account, fraud_type):
        self.labels.setdefault(account, set()).add(fraud_type)

    def random_time(self, day_min=0, day_max=DAYS - 1):
        return START + timedelta(days=self.rng.randint(day_min, day_max),
                                 seconds=self.rng.randint(0, 86399))

    def other_person(self, *exclude):
        while True:
            p = self.rng.choice(self.people)
            if p not in exclude:
                return p

    def normal_activity(self):
        contacts = {p: self.rng.sample(self.people, self.rng.randint(3, 8)) for p in self.people}
        for _ in range(self.n_normal):
            src = self.rng.choice(self.people)
            r = self.rng.random()
            if r < 0.5:
                dst, amount = self.rng.choice(self.merchants), self.np_rng.lognormal(6, 0.8)
            elif r < 0.9:
                dst, amount = self.rng.choice(contacts[src]), self.np_rng.lognormal(7, 1.0)
            else:
                dst, amount = self.rng.choice(self.people), self.np_rng.lognormal(7, 1.2)
            if dst != src:
                self.add(src, dst, min(amount, 40000), self.random_time())
        for m in self.merchants:
            for _ in range(self.rng.randint(5, 15)):
                self.add(m, self.rng.choice(self.people), self.np_rng.lognormal(8.5, 0.5), self.random_time())

    def money_loop(self, size):
        ring = self.rng.sample(self.people, size)
        amount = self.rng.uniform(8000, 45000) if self.hard else self.rng.uniform(20000, 45000)
        when = self.random_time(5, DAYS - 5)
        for i in range(size):
            self.add(ring[i], ring[(i + 1) % size], amount, when, "loop")
            amount *= self.rng.uniform(0.85, 0.99) if self.hard else self.rng.uniform(0.97, 0.995)
            when += timedelta(hours=self.rng.uniform(1, 30) if self.hard else self.rng.uniform(1, 8))
        for a in ring:
            self.label(a, "loop")

    def smurfing(self, n_smurfs=12):
        mule = self.rng.choice(self.people)
        boss = self.other_person(mule)
        when = self.random_time(5, DAYS - 5)
        total = 0
        for s in self.rng.sample(self.people, n_smurfs):
            if s in (mule, boss):
                continue
            amt = self.rng.uniform(38000, 49900) if self.hard else self.rng.uniform(45000, 49900)
            self.add(s, mule, amt, when, "smurfing")
            total += amt
            when += timedelta(hours=self.rng.uniform(0.5, 24) if self.hard else self.rng.uniform(0.5, 6))
        self.add(mule, boss, total * 0.95, when + timedelta(hours=2), "smurfing")
        self.label(mule, "smurfing")

    def layering(self, length):
        chain = self.rng.sample(self.people, length)
        amount = self.rng.uniform(30000, 80000)
        when = self.random_time(5, DAYS - 5)
        for a, b in zip(chain, chain[1:]):
            self.add(a, b, amount, when, "layering")
            amount *= self.rng.uniform(0.80, 0.99) if self.hard else self.rng.uniform(0.95, 0.99)
            when += timedelta(minutes=self.rng.uniform(30, 400) if self.hard else self.rng.uniform(10, 120))
        for a in chain[1:-1]:
            self.label(a, "layering")

    def burst(self, n=15):
        acc = self.rng.choice(self.people)
        when = self.random_time(5, DAYS - 5)
        for _ in range(n):
            self.add(acc, self.other_person(acc), self.rng.uniform(2000, 9000), when, "burst")
            when += timedelta(minutes=self.rng.uniform(2, 12) if self.hard else self.rng.uniform(1, 6))
        self.label(acc, "burst")

    def build(self, n_each=3):
        self.normal_activity()
        for _ in range(n_each):
            self.money_loop(self.rng.randint(3, 6))
            self.smurfing()
            self.layering(self.rng.randint(4, 7))
            self.burst()
        df = pd.DataFrame(self.rows).sort_values("timestamp").reset_index(drop=True)
        df.insert(0, "tx_id", [f"TX{i:06d}" for i in range(len(df))])
        labels = pd.DataFrame(
            [(a, ";".join(sorted(t))) for a, t in sorted(self.labels.items())],
            columns=["account", "fraud_type"],
        )
        return df, labels


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic transactions")
    parser.add_argument("--accounts", type=int, default=500)
    parser.add_argument("--transactions", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data")
    parser.add_argument("--hard", action="store_true", help="smarter, harder-to-catch fraud")
    args = parser.parse_args()

    df, labels = Generator(args.accounts, args.transactions, args.seed, args.hard).build()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    df.to_csv(out / "transactions.csv", index=False)
    labels.to_csv(out / "labels.csv", index=False)

    print(f"Transactions: {len(df)}  (fraud: {(df['fraud_type'] != '').sum()})")
    print(f"Accounts: {pd.concat([df['source_account'], df['target_account']]).nunique()}")
    print("Fraud accounts by type:")
    print(labels["fraud_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
