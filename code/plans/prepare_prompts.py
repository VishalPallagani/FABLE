import os
import glob
import pandas as pd

CSV_DIR     = "../benchmarks"    # your pddl_*.csv files
PROMPT_DIR  = "../prompts"       # where enriched CSVs will go
N_SAMPLES   = 100
RANDOM_SEED = 42

os.makedirs(PROMPT_DIR, exist_ok=True)

definitions = {
    "Reaching Definitions":    "Tracks which definitions (assignments) of variables can reach a particular program point without being overridden.",
    "Very Busy Expressions":    "An expression is very busy if on every path forward from that point, the expression will be evaluated before any operand changes.",
    "Available Expressions":    "An expression is available if it has already been computed and its operands have not been redefined since.",
    "Live Variable Analysis":  "A variable is live if its value will be used again in the future (i.e., before being overwritten or discarded).",
    "Interval Analysis":       "Determines the possible numeric ranges (e.g., bounds on variables) at each point in a program.",
    "Type-State Analysis":     "Ensures an object is used only in valid states (e.g., a file must be opened before reading).",
    "Taint Analysis":          "Tracks tainted or untrusted data to ensure it does not reach sensitive areas without sanitization.",
    "Concurrency Analysis":    "Examines parallel (concurrent) execution paths to detect data races or synchronization issues."
}

def make_prompt(row, bench_name, definition):
    steps = [s.strip() for s in str(row["plan"]).split(" ; ") if s.strip()]
    plan_block = "\n".join(steps)
    return (
        "You are an AI analyst being evaluated on data-flow analyses for planning tasks.\n"
        f"Benchmark: {bench_name}\n"
        f"Definition: {definition}\n\n"
        "Given the following planning task:\n\n"
        f"  Goal: {row['goal']}\n\n"
        "  Plan steps:\n"
        f"{plan_block}\n\n"
        "Question:\n"
        f"  {row['question']}\n\n"
        "Please answer with only “Yes” or “No” without any further explanation. Adhere to this rule strictly."
    )

for csv_path in glob.glob(os.path.join(CSV_DIR, "pddl_*.csv")):
    fname      = os.path.basename(csv_path)
    safe_name  = fname[len("pddl_"):-len(".csv")]
    bench_name = safe_name.replace("_", " ")
    definition = definitions[bench_name]

    # load, drop rows without an answer, shuffle, take top N
    df = pd.read_csv(csv_path)
    df = df[df["answer"].notna() & df["answer"].astype(str).str.strip().astype(bool)]
    df = df.sample(frac=1, random_state=RANDOM_SEED).head(N_SAMPLES).reset_index(drop=True)

    # append prompt column
    df["prompt"] = df.apply(lambda row: make_prompt(row, bench_name, definition), axis=1)

    # write out
    out_path = os.path.join(PROMPT_DIR, fname)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows)")
