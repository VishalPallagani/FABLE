# prepare_prompts.py

import os
import glob
import re
import pandas as pd

# === Configuration ===
CSV_DIR     = "../../benchmarks/recipe"   # folder with your recipe_<benchmark>.csv files
PROMPT_DIR  = "../../prompts/recipe"      # where enriched CSVs will go
N_SAMPLES   = 100
RANDOM_SEED = 42

os.makedirs(PROMPT_DIR, exist_ok=True)

# Definitions for each benchmark
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

# Precompiled regex to extract the prompt components
PROMPT_RE = re.compile(
    r"Goal:\s*(?P<goal>.*?)\r?\n\s*\r?\n"
    r"Steps:\r?\n(?P<steps>.*?)\r?\n\s*\r?\n"
    r"Question:\r?\n(?P<question>.*)",
    re.S
)

# Regex to split the steps_block into individual "Step X: ..." entries
STEP_RE = re.compile(
    r"(Step\s*\d+:\s*.*?)"           # capture "Step N: ..." up to...
    r"(?=(?:\r?\nStep\s*\d+:)|\Z)",  # ...the next "Step M:" or end of block
    re.S
)

def parse_raw_prompt(raw: str):
    """
    Extract goal, list of steps, and question from the raw 'prompt' column.
    Uses regex to find the Goal, Steps block, and Question, then splits
    Steps into discrete entries based on "Step <number>:" markers.
    """
    m = PROMPT_RE.search(raw)
    if not m:
        raise ValueError(f"Could not parse prompt block:\n{raw[:200]}...")
    goal = m.group("goal").strip()
    steps_block = m.group("steps").strip()
    question = m.group("question").strip()

    # Use STEP_RE to extract each full step (including any line breaks)
    raw_steps = STEP_RE.findall(steps_block)
    # Normalize whitespace within each step
    steps = [re.sub(r"\s+", " ", s).strip() for s in raw_steps]

    return goal, steps, question

def make_prompt(goal, steps, question, bench_name, definition):
    """
    Construct the enriched prompt text given parsed components,
    injecting the benchmark name and definition, plus answer instructions.
    """
    header = (
        "You are an AI analyst being evaluated on data-flow analyses for cooking recipes.\n"
        f"Benchmark: {bench_name}\n"
        f"Definition: {definition}\n\n"
        "Given the following recipe scenario:\n\n"
        f"  Goal: {goal}\n\n"
        "  Steps:\n"
        + "\n".join(f"  {s}" for s in steps)
        + "\n\n"
        "Question:\n"
        f"  {question}\n\n"
    )
    if bench_name == "Interval Analysis":
        return header + "Please provide the numeric interval exactly as it appears (e.g., “1/2 hour”)."
    else:
        return header + 'Please answer with only "Yes" or "No" without any further explanation.'

# === Main processing loop ===
for csv_path in glob.glob(os.path.join(CSV_DIR, "recipe_*.csv")):
    fname      = os.path.basename(csv_path)                 # e.g. "recipe_available_expressions.csv"
    safe_name  = os.path.splitext(fname)[0]                 # "recipe_available_expressions"
    key        = safe_name[len("recipe_"):]                 # "available_expressions"
    bench_name = key.replace("_", " ").title()             # "Available Expressions"

    if bench_name not in definitions:
        print(f"[skip] no definition for '{bench_name}' in '{fname}'")
        continue
    definition = definitions[bench_name]

    # Load the raw prompt/answer CSV
    df = pd.read_csv(csv_path)
    if "prompt" not in df.columns or "answer" not in df.columns:
        print(f"[error] '{fname}' missing required columns 'prompt'/'answer'")
        continue

    # Filter out rows without an answer
    df = df[df["answer"].notna() & df["answer"].astype(str).str.strip().astype(bool)]
    # Shuffle and sample top N
    df = df.sample(frac=1, random_state=RANDOM_SEED).head(N_SAMPLES).reset_index(drop=True)

    # Parse and regenerate enriched prompts, and assemble output columns
    output_rows = []
    for _, row in df.iterrows():
        goal, steps, question = parse_raw_prompt(row["prompt"])
        plan = repr(steps)  # Python-list literal, e.g. "['Step 1: ...', 'Step 2: ...']"
        enriched_prompt = make_prompt(goal, steps, question, bench_name, definition)
        output_rows.append({
            "goal":     goal,
            "plan":     plan,
            "question": question,
            "answer":   row["answer"],
            "prompt":   enriched_prompt
        })

    out_df = pd.DataFrame(output_rows, columns=["goal","plan","question","answer","prompt"])

    # Write to prompts directory
    out_path = os.path.join(PROMPT_DIR, fname)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df)} rows)")
