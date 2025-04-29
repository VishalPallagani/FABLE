import os
import glob
import pandas as pd

CSV_DIR     = "../../benchmarks/travel_routes"   # folder with your travel_<benchmark>.csv files
PROMPT_DIR  = "../../prompts/travel_routes"      # where enriched CSVs will go
N_SAMPLES   = 100
RANDOM_SEED = 42

os.makedirs(PROMPT_DIR, exist_ok=True)

definitions = {
    "Reaching Definitions":    "Tracks which definitions (assignments) of variables can reach a particular program point without being overridden.",
    "Very Busy Expressions":   "An expression is very busy if on every path forward from that point, the expression will be evaluated before any operand changes.",
    "Available Expressions":   "An expression is available if it has already been computed and its operands have not been redefined since.",
    "Live Variable Analysis":  "A variable is live if its value will be used again in the future (i.e., before being overwritten or discarded).",
    "Interval Analysis":       "Determines the possible numeric ranges (e.g., bounds on variables) at each point in a program.",
    "Type State Analysis":     "Ensures an object is used only in valid states (e.g., a file must be opened before reading).",
    "Taint Analysis":          "Tracks tainted or untrusted data to ensure it does not reach sensitive areas without sanitization.",
    "Concurrency Analysis":    "Examines parallel (concurrent) execution paths to detect data races or synchronization issues."
}

def make_prompt(row, bench_name, definition):
    steps = [s.strip() for s in str(row["plan"]).split(" ; ") if s.strip()]
    plan_block = "\n".join(steps)

    header = (
        "You are an AI analyst being evaluated on data-flow analyses for travel routes.\n"
        f"Benchmark: {bench_name}\n"
        f"Definition: {definition}\n\n"
        "Given the following travel planning scenario:\n\n"
        f"  Goal: {row['goal']}\n\n"
        "  Route steps:\n"
        f"{plan_block}\n\n"
        "Question:\n"
        f"  {row['question']}\n\n"
    )

    if bench_name == "Interval Analysis":
        return header + \
            "Please provide the numeric interval in the format [x, y] without any further explanation."
    else:
        return header + \
            "Please answer with only “Yes” or “No” without any further explanation. Adhere to this rule strictly."

for csv_path in glob.glob(os.path.join(CSV_DIR, "travel_*.csv")):
    fname      = os.path.basename(csv_path)                # e.g. "travel_available_expressions.csv"
    safe_name  = os.path.splitext(fname)[0]                # "travel_available_expressions"

    # strip leading "travel_" and title-case to match definitions keys
    key        = safe_name[len("travel_"):]
    bench_name = key.replace("_", " ").title()            # "Available Expressions"

    if bench_name not in definitions:
        print(f"No definition for '{bench_name}', skipping {fname}")
        continue
    definition = definitions[bench_name]

    # load and normalize columns
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "Goal":         "goal",
        "Plan":         "plan",
        "Question":     "question",
        "Ground Truth": "answer"
    })[["goal", "plan", "question", "answer"]]

    # filter out rows without an answer
    df = df[df["answer"].notna() & df["answer"].astype(str).str.strip().astype(bool)]

    # shuffle and sample top N
    df = df.sample(frac=1, random_state=RANDOM_SEED).head(N_SAMPLES).reset_index(drop=True)

    # generate the prompt column
    df["prompt"] = df.apply(lambda r: make_prompt(r, bench_name, definition), axis=1)

    # reorder and write out
    out_df = df[["goal", "plan", "question", "answer", "prompt"]]
    out_path = os.path.join(PROMPT_DIR, fname)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df)} rows)")
