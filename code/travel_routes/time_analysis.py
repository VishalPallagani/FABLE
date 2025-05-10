import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to timing CSV (model,benchmark,seconds)
CSV_PATH     = "../../results/new_travel_routes/timing.log"
# Path where to save the combined visualization
OUTPUT_IMAGE = "../../scores/new_travel_routes/travel_route_model_timing_comparison.png"

# mapping file‐internal model names → display names
alias_map = {
    "deepseek_r1_8b":  "deepseek-r1:8b",
    "llama3_1":        "llama3.1:8b",
    "granite_code_8b": "granite-code:8b",
}

# desired benchmark order for the rows
benchmarks_order = [
    "reaching_definitions", "very_busy_expressions",
    "available_expressions", "live_variable_analysis",
    "interval_analysis", "type_state_analysis",
    "taint_analysis", "concurrency_analysis"
]

df = pd.read_csv(CSV_PATH)
# pivot so that each row is a benchmark, each column a model
pivot = df.pivot(index="benchmark", columns="model", values="seconds")
pivot = pivot.reindex(benchmarks_order)
pivot = pivot.rename(columns=alias_map)

# compute average inference time per model
avg_times = pivot.mean(axis=0)

# slowdown = T_deepseek / T_other
slowdown = pd.DataFrame({
    "vs llama3.1:8b":   pivot["deepseek-r1:8b"] / pivot["llama3.1:8b"],
    "vs granite-code:8b": pivot["deepseek-r1:8b"] / pivot["granite-code:8b"]
}, index=pivot.index)

print("\nPer‐benchmark slowdown of deepseek-r1:8b:")
print(slowdown.round(2))
print("\nAverage slowdown:")
print(f"  deepseek vs llama3.1:8b   = {slowdown['vs llama3.1:8b'].mean():.2f}×")
print(f"  deepseek vs granite-code:8b = {slowdown['vs granite-code:8b'].mean():.2f}×\n")

fig, (ax_heat, ax_bar) = plt.subplots(2, 1, figsize=(12, 14),
                                      gridspec_kw={"height_ratios":[3, 1]})

# --- Heatmap of log10 times ---
log_data = np.log10(pivot.values)
im = ax_heat.imshow(log_data, aspect="auto", cmap="viridis")
models = pivot.columns.tolist()
bench_labels = [b.replace("_", " ").title() for b in pivot.index]

ax_heat.set_xticks(np.arange(len(models)))
ax_heat.set_xticklabels(models, rotation=45, ha="right", fontsize=12)
ax_heat.set_yticks(np.arange(len(bench_labels)))
ax_heat.set_yticklabels(bench_labels, fontsize=12)
ax_heat.set_title("Inference Latency Heatmap (Travel Routes)", fontsize=16, pad=20)

# annotate each cell with the raw seconds
mean_log = log_data.mean()
for i in range(len(bench_labels)):
    for j in range(len(models)):
        val = pivot.iloc[i, j]
        color = "white" if log_data[i, j] > mean_log else "black"
        ax_heat.text(j, i, f"{val:.0f}s", ha="center", va="center",
                     color=color, fontsize=10, fontweight="bold")

# colorbar
cbar = fig.colorbar(im, ax=ax_heat, pad=0.02)
cbar.set_label("log₁₀(Time [s] for 100 prompts)", fontsize=12)

# --- Bar chart of average times ---
x = np.arange(len(avg_times))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # distinct colors
ax_bar.bar(x, avg_times, color=colors, width=0.6)
ax_bar.set_xticks(x)
ax_bar.set_xticklabels(avg_times.index, rotation=45, ha="right", fontsize=12)
ax_bar.set_yscale("log")
ax_bar.set_ylabel("Average Time [s] (log scale)", fontsize=14)
ax_bar.set_title("Average Inference Time per Model (100 prompts)", fontsize=16)

# annotate bar values on top
for i, t in enumerate(avg_times):
    ax_bar.text(i, t * 1.05, f"{t:.1f}s", ha="center", va="bottom",
                fontsize=12, fontweight="bold")

plt.tight_layout()
os.makedirs(os.path.dirname(OUTPUT_IMAGE), exist_ok=True)
plt.savefig(OUTPUT_IMAGE, dpi=300)
plt.show()