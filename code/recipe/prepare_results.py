# run_recipe_evaluation.py

import os
import glob
import time
import pandas as pd
import ollama

# === Configuration ===
NEW_CSV_DIR = "../../prompts/recipe/"   # where your recipe_*.csv prompts live
RESULT_DIR  = "../../results/recipe/"   # where result_*.csv and timing.log will go

MODELS = {
    "deepseek_r1_8b":  "deepseek-r1:8b",
    "llama3_1":        "llama3.1",
    "granite_code_8b": "granite-code:8b",
}

os.makedirs(RESULT_DIR, exist_ok=True)
log_path = os.path.join(RESULT_DIR, "timing.log")

# clear previous log
with open(log_path, "w") as f:
    f.write("model,benchmark,seconds\n")

for csv_path in glob.glob(os.path.join(NEW_CSV_DIR, "recipe_*.csv")):
    fname   = os.path.basename(csv_path)                   # e.g. "recipe_available_expressions.csv"
    safe_bm = fname[len("recipe_"):-len(".csv")]           # "available_expressions"

    df = pd.read_csv(csv_path)

    for alias, model_ref in MODELS.items():
        records = []
        print(f"▶ Querying {model_ref} on {safe_bm} ({len(df)} prompts)…")

        start = time.time()
        for _, row in df.iterrows():
            prompt       = row["prompt"]
            ground_truth = row["answer"]

            try:
                stream = ollama.chat(
                    model=model_ref,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                response = "".join(chunk["message"]["content"] for chunk in stream)
            except Exception as e:
                response = f"ERROR: {e}"

            records.append({
                "prompt":       prompt,
                "ground_truth": ground_truth,
                "response":     response
            })

        elapsed = time.time() - start

        # log timing
        with open(log_path, "a") as f:
            f.write(f"{alias},{safe_bm},{elapsed:.2f}\n")
        print(f"✓ {model_ref} on {safe_bm} took {elapsed:.2f}s")

        # save results
        result_df = pd.DataFrame(records, columns=["prompt", "ground_truth", "response"])
        out_name  = f"result_{alias}_{safe_bm}.csv"
        result_df.to_csv(os.path.join(RESULT_DIR, out_name), index=False)
        print(f"✓ Wrote {out_name}\n")
