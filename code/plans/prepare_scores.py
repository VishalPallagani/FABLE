import os
import glob
import pandas as pd

RESULT_DIR = "../results"   # where your result_*.csv files live
SCORES_DIR   = "../scores"                 # where you want to save the summary CSV
SCORES_FILE  = os.path.join(SCORES_DIR, "overall_performance.csv")

safe_benchmarks = [
    "Reaching_Definitions",
    "Very_Busy_Expressions",
    "Available_Expressions",
    "Live_Variable_Analysis",
    "Interval_Analysis",
    "Type-State_Analysis",
    "Taint_Analysis",
    "Concurrency_Analysis",
]

display_names = {
    "Reaching_Definitions":    "Reaching Definitions",
    "Very_Busy_Expressions":   "Very Busy Expressions",
    "Available_Expressions":   "Available Expressions",
    "Live_Variable_Analysis":  "Live Variable Analysis",
    "Interval_Analysis":       "Interval Analysis",
    "Type-State_Analysis":     "Type-State Analysis",
    "Taint_Analysis":          "Taint Analysis",
    "Concurrency_Analysis":    "Concurrency Analysis"
}

os.makedirs(SCORES_DIR, exist_ok=True)

scores = {}

for path in glob.glob(os.path.join(RESULT_DIR, "result_*_*.csv")):
    fname = os.path.basename(path)  
    base  = fname[len("result_"):-len(".csv")]

    bench_key = None
    model_alias = None
    for sb in safe_benchmarks:
        if base.endswith(sb):
            bench_key   = sb
            model_alias = base[: -(len(sb) + 1)]
            break
    if bench_key is None:
        print(f"Skipping unrecognized file: {fname}")
        continue

    df = pd.read_csv(path)
    correct_count = df.apply(
        lambda r: str(r["ground_truth"]).strip().lower() in r["response"].lower(),
        axis=1
    ).sum()

    scores.setdefault(model_alias, {})[bench_key] = int(correct_count)

rows = []
for model_alias, bench_dict in scores.items():
    row = {"LLM": model_alias}
    for sb in safe_benchmarks:
        row[display_names[sb]] = bench_dict.get(sb, 0)
    rows.append(row)

perf_df = pd.DataFrame(rows).set_index("LLM")[list(display_names.values())]

perf_df.to_csv(SCORES_FILE)
print(f"Wrote overall performance table to {SCORES_FILE}")