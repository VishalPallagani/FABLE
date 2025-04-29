import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
CSV_PATH     = "../scores/plans/overall_performance.csv"
OUTPUT_IMAGE = "../results/plans/plans_radar.png"

# Mapping from internal alias to display name
alias_map = {
    "deepseek_r1_8b":  "deepseek-r1:8b",
    "granite_code_8b": "granite-code:8b",
    "llama3_1":        "llama3.1:8b",  # as requested
}

# ─── LOAD & PREPARE ─────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
# rename the index (LLM column) to the desired display names
df = df.rename(columns={df.columns[0]: "LLM"})
df["LLM"] = df["LLM"].map(alias_map).fillna(df["LLM"])
df = df.set_index("LLM")

labels   = df.columns.tolist()
n_labels = len(labels)
angles   = np.linspace(0, 2 * np.pi, n_labels, endpoint=False).tolist()
angles  += angles[:1]

# ─── PLOT ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for llm, row in df.iterrows():
    values = row.values.tolist()
    values += values[:1]
    ax.plot(angles, values, label=llm)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10)
ax.set_yticks(range(0, 101, 20))
ax.set_yticklabels([f"{i}%" for i in range(0, 101, 20)])
ax.set_ylim(0, 100)

ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
plt.title("LLM Performance on Automated Plans Benchmarks", pad=20)

# ─── SAVE ──────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
plt.savefig(OUTPUT_IMAGE, bbox_inches="tight")
plt.show()
