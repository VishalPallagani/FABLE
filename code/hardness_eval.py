import matplotlib.pyplot as plt
import numpy as np

# Raw results you shared
plans_results = {
    "random": [69, 54, 47, 54, 57, 0, 56, 49],
    "deepseek_r1_8b": [92, 97, 87, 97, 31, 94, 100, 63],
    "granite_code_8b": [55, 84, 89, 52, 16, 47, 27, 62],
    "llama3_1": [67, 79, 55, 56, 8, 51, 42, 52],
}

travel_results = {
    "random": [56, 53, 55, 54, 0, 69, 52, 61],
    "deepseek_r1_8b": [92, 100, 34, 94, 57, 99, 70, 100],
    "granite_code_8b": [46, 36, 90, 47, 0, 63, 49, 52],
    "llama3_1": [67, 59, 29, 46, 4, 84, 34, 88],
}

models = ["deepseek_r1_8b", "granite_code_8b", "llama3_1"]

def compute_average_and_std(results):
    avg = {}
    std = {}
    for model in models + ["random"]:
        avg[model] = np.mean(results[model])
        std[model] = np.std(results[model])
    return avg, std

plans_avg, plans_std = compute_average_and_std(plans_results)
travel_avg, travel_std = compute_average_and_std(travel_results)

# Compute Delta over Random
plans_delta = {model: plans_avg[model] - plans_avg["random"] for model in models}
travel_delta = {model: travel_avg[model] - travel_avg["random"] for model in models}

# Prepare data
domains = ["Plans", "Travel Routes"]
x = np.arange(len(domains))  # label locations
width = 0.25  # bar width

# Bar heights
deepseek_delta = [plans_delta["deepseek_r1_8b"], travel_delta["deepseek_r1_8b"]]
granite_delta  = [plans_delta["granite_code_8b"], travel_delta["granite_code_8b"]]
llama_delta    = [plans_delta["llama3_1"], travel_delta["llama3_1"]]

# Error bars (standard deviation over 8 tasks)
deepseek_std = [plans_std["deepseek_r1_8b"], travel_std["deepseek_r1_8b"]]
granite_std  = [plans_std["granite_code_8b"], travel_std["granite_code_8b"]]
llama_std    = [plans_std["llama3_1"], travel_std["llama3_1"]]

# Plot
fig, ax = plt.subplots(figsize=(10,6))

rects1 = ax.bar(x - width, deepseek_delta, width, label='deepseek-r1:8B', color='royalblue', yerr=deepseek_std, capsize=5)
rects2 = ax.bar(x, granite_delta, width, label='granite-code:8B', color='seagreen', yerr=granite_std, capsize=5)
rects3 = ax.bar(x + width, llama_delta, width, label='llama3.1:8B', color='salmon', yerr=llama_std, capsize=5)

# Labels and title
ax.set_ylabel('Delta over Random Baseline (%)')
ax.set_title('Domain-Wise Delta over Random Accuracy (± Std Dev)')
ax.set_xticks(x)
ax.set_xticklabels(domains)
ax.axhline(0, color='gray', linewidth=0.8)  # horizontal line at 0
ax.legend()

# Optional: Add value labels
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

for rects in [rects1, rects2, rects3]:
    autolabel(rects)

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.show()
