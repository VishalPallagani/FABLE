import os
import glob
import re
import pandas as pd

RESULT_DIR  = "../../results/new_travel_routes"    # folder with your result_<model>_<bench>.csv
SCORES_DIR  = "../../scores/new_travel_routes"     # where to write overall_performance.csv
SCORES_FILE = os.path.join(SCORES_DIR, "overall_performance.csv")

# lowercase “safe” keys matching the suffix of filenames
safe_benchmarks = [
    "reaching_definitions",
    "very_busy_expressions",
    "available_expressions",
    "live_variable_analysis",
    "interval_analysis",
    "type_state_analysis",
    "taint_analysis",
    "concurrency_analysis",
]

# human‐friendly column names
display_names = {
    "reaching_definitions":       "Reaching Definitions",
    "very_busy_expressions":      "Very Busy Expressions",
    "available_expressions":      "Available Expressions",
    "live_variable_analysis":     "Live Variable Analysis",
    "interval_analysis":          "Interval Analysis",
    "type_state_analysis":        "Type-State Analysis",
    "taint_analysis":             "Taint Analysis",
    "concurrency_analysis":       "Concurrency Analysis"
}

os.makedirs(SCORES_DIR, exist_ok=True)

scores = {}

for path in glob.glob(os.path.join(RESULT_DIR, "result_*_*.csv")): 
    
    fname = os.path.basename(path)
    # strip “result_” prefix and “.csv” suffix
    base  = fname[len("result_"):-len(".csv")]
    base_lower = base.lower()

    # identify benchmark key & model alias by suffix matching
    bench_key = None
    model_alias = None
    for sb in safe_benchmarks:
        if base_lower.endswith(sb):
            bench_key   = sb
            # model_alias is everything before the underscore + sb
            model_alias = base[:- (len(sb) + 1)]
            break
    if bench_key is None:
        print(f"Skipping unrecognized file: {fname}")
        continue

    # load the results
    df = pd.read_csv(path)

    def is_correct(row):
        gt   = str(row["ground_truth"]).strip()
        resp = str(row["response"]).lower()

        if bench_key == "interval_analysis":
            # ground_truth format: "[x, y]"
            m = re.match(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", gt)
            if not m:
                return False
            x, y = m.group(1), m.group(2)
            return (x in resp) and (y in resp)
        else:
            return gt.lower() in resp

    correct_count = df.apply(is_correct, axis=1).sum()
    scores.setdefault(model_alias, {})[bench_key] = int(correct_count)

# build the summary table
rows = []
for model_alias, bench_dict in scores.items():
    row = {"LLM": model_alias}
    for sb in safe_benchmarks:
        row[display_names[sb]] = bench_dict.get(sb, 0)
    rows.append(row)

perf_df = pd.DataFrame(rows).set_index("LLM")[list(display_names.values())]

# write it out
perf_df.to_csv(SCORES_FILE)
print(f"Wrote overall performance table to {SCORES_FILE}")
print(perf_df)
