import os
import glob
import re
import random
import pandas as pd

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
random.seed(42)

TRAVEL_PROMPT_DIR   = "../prompts/new_travel_routes"
PLANNING_PROMPT_DIR = "../prompts/plans"
RECIPE_PROMPT_DIR   = "../prompts/recipe"

RESULT_DIR_TRAVEL   = "../results/new_baseline_travel"
RESULT_DIR_PLAN     = "../results/baseline_plan"
RESULT_DIR_RECIPE   = "../results/baseline_recipe"

SUMMARY_PATH        = "../scores/ne_recipe_baseline_random_performance.csv"

os.makedirs(RESULT_DIR_TRAVEL, exist_ok=True)
os.makedirs(RESULT_DIR_PLAN,   exist_ok=True)
os.makedirs(RESULT_DIR_RECIPE, exist_ok=True)
os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)

# Benchmarks identifiers
benchmarks = [
    "reaching_definitions",
    "very_busy_expressions",
    "available_expressions",
    "live_variable_analysis",
    "interval_analysis",
    "type_state_analysis",
    "taint_analysis",
    "concurrency_analysis",
]

display_names = {
    "reaching_definitions":    "Reaching Definitions",
    "very_busy_expressions":   "Very Busy Expressions",
    "available_expressions":   "Available Expressions",
    "live_variable_analysis":  "Live Variable Analysis",
    "interval_analysis":       "Interval Analysis",
    "type_state_analysis":     "Type-State Analysis",
    "taint_analysis":          "Taint Analysis",
    "concurrency_analysis":    "Concurrency Analysis"
}

scores = {
    "Random Travel": {},
    "Random Plan": {},
    "Random Recipe": {}
}

# ─── PRECOMPUTE TRAVEL INTERVAL RANGE ────────────────────────────────────────────
interval_file = os.path.join(TRAVEL_PROMPT_DIR, "travel_interval_analysis.csv")
min_gt, max_gt = 0, 100
if os.path.exists(interval_file):
    df_int = pd.read_csv(interval_file)
    ints = []
    for ans in df_int["answer"].dropna().astype(str):
        m = re.match(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", ans)
        if m:
            ints.append((int(m.group(1)), int(m.group(2))))
    if ints:
        min_gt = min(x for x, _ in ints)
        max_gt = max(y for _, y in ints)

# ─── PRECOMPUTE RECIPE INTERVAL ANSWERS ───────────────────────────────────────────
rec_interval_file = os.path.join(RECIPE_PROMPT_DIR, "recipe_interval_analysis.csv")
recipe_interval_answers = []
if os.path.exists(rec_interval_file):
    df_rec_int = pd.read_csv(rec_interval_file)
    recipe_interval_answers = (
        df_rec_int["answer"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

# ─── RANDOM BASELINE & EVALUATION FOR TRAVEL ────────────────────────────────────
for bench in benchmarks:
    pattern = f"travel_{bench}.csv"
    for path in glob.glob(os.path.join(TRAVEL_PROMPT_DIR, pattern)):
        df = pd.read_csv(path)
        records = []
        correct = 0

        for _, row in df.iterrows():
            gt = str(row["answer"]).strip()
            if bench != "interval_analysis":
                resp = random.choice(["Yes", "No"])
                is_corr = (resp.lower() == gt.lower())
            else:
                low  = random.randint(min_gt, max_gt)
                high = random.randint(low, max_gt)
                resp = f"[{low}, {high}]"
                m = re.match(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", gt)
                if m:
                    x, y = m.group(1), m.group(2)
                    is_corr = (x in resp) and (y in resp)
                else:
                    is_corr = False

            records.append({
                "prompt":       row["prompt"],
                "ground_truth": gt,
                "response":     resp
            })
            correct += int(is_corr)

        out_name = f"result_random_travel_{bench}.csv"
        pd.DataFrame(records)[["prompt","ground_truth","response"]].to_csv(
            os.path.join(RESULT_DIR_TRAVEL, out_name), index=False
        )
        scores["Random Travel"][bench] = correct

# ─── RANDOM BASELINE & EVALUATION FOR PLANNING ─────────────────────────────────
for bench in benchmarks:
    pattern = f"pddl_{bench}.csv"
    for path in glob.glob(os.path.join(PLANNING_PROMPT_DIR, pattern)):
        df = pd.read_csv(path)
        records = []
        correct = 0

        for _, row in df.iterrows():
            gt = str(row["answer"]).strip()
            if bench != "interval_analysis":
                resp = random.choice(["Yes", "No"])
                is_corr = (resp.lower() == gt.lower())
            else:
                steps = [s.strip() for s in str(row["plan"]).split(" ; ") if s.strip()]
                n = len(steps)
                choice = random.choice(["Before", "After", "Between"])
                if choice == "Before":
                    i = random.randint(1, max(1, n))
                    resp = f"Before Step {i}"
                elif choice == "After":
                    i = random.randint(1, max(1, n))
                    resp = f"After Step {i}"
                else:
                    if n < 2:
                        resp = "After Step 1"
                    else:
                        i = random.randint(1, n-1)
                        j = random.randint(i+1, n)
                        resp = f"Between Step {i} and Step {j}"
                is_corr = (resp.lower() == gt.lower())

            records.append({
                "prompt":       row["prompt"],
                "ground_truth": gt,
                "response":     resp
            })
            correct += int(is_corr)

        out_name = f"result_random_plan_{bench}.csv"
        pd.DataFrame(records)[["prompt","ground_truth","response"]].to_csv(
            os.path.join(RESULT_DIR_PLAN, out_name), index=False
        )
        scores["Random Plan"][bench] = correct

# ─── RANDOM BASELINE & EVALUATION FOR RECIPE ────────────────────────────────────
for bench in benchmarks:
    pattern = f"recipe_{bench}.csv"
    for path in glob.glob(os.path.join(RECIPE_PROMPT_DIR, pattern)):
        df = pd.read_csv(path)
        records = []
        correct = 0

        for _, row in df.iterrows():
            gt = str(row["answer"]).strip()
            if bench != "interval_analysis":
                resp = random.choice(["Yes", "No"])
                is_corr = (resp.lower() == gt.lower())
            else:
                # pick a random actual interval answer
                if recipe_interval_answers:
                    resp = random.choice(recipe_interval_answers)
                else:
                    resp = gt  # fallback to ground truth
                is_corr = (resp.lower() == gt.lower())

            records.append({
                "prompt":       row["prompt"],
                "ground_truth": gt,
                "response":     resp
            })
            correct += int(is_corr)

        out_name = f"result_random_recipe_{bench}.csv"
        pd.DataFrame(records)[["prompt","ground_truth","response"]].to_csv(
            os.path.join(RESULT_DIR_RECIPE, out_name), index=False
        )
        scores["Random Recipe"][bench] = correct

# ─── BUILD & SAVE SUMMARY TABLE ─────────────────────────────────────────────────
rows = []
for model_name, bench_dict in scores.items():
    row = {"Model": model_name}
    for sb in benchmarks:
        row[display_names[sb]] = bench_dict.get(sb, 0)
    rows.append(row)

perf_df = pd.DataFrame(rows).set_index("Model")[list(display_names.values())]
perf_df.to_csv(SUMMARY_PATH)
print("Baseline performance saved to", SUMMARY_PATH)
print(perf_df)
