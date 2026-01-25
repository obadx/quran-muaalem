import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_json("results/summary.jsonl", lines=True)

df.to_csv("results/final_report.csv", index=False)

df.groupby(["model", "audio_sec", "batch"])[
    ["avg", "p95", "p99", "throughput"]
].mean().plot(subplots=True, figsize=(12, 8))

plt.tight_layout()
plt.savefig("results/aggregate_metrics.png")
