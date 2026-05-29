import pandas as pd

w = pd.read_csv("results/latest/weights.csv")

for strategy in w["strategy"].unique():
    subset = w[w["strategy"] == strategy]

    last_rebalance = subset["rebalance_date"].max()

    current = subset[subset["rebalance_date"] == last_rebalance]

    print(
        strategy,
        (current["weight"] > 1e-6).sum()
    )