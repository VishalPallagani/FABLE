import os
import glob
import pandas as pd
import ollama

NEW_CSV_DIR = "../prompts"     # where your pddl_*.csv (with prompt) live
RESULT_DIR  = "../results"      # where result_*.csv files should go

# Ollama models to query: alias → model reference
MODELS = {
    "deepseek_r1_8b":  "deepseek-r1:8b",
    "llama3_1":        "llama3.1",
    "granite_code_8b": "granite-code:8b",
}
os.makedirs(RESULT_DIR, exist_ok=True)
for csv_path in glob.glob(os.path.join(NEW_CSV_DIR, "pddl_*.csv")):
    fname      = os.path.basename(csv_path)
    safe_bm    = fname[len("pddl_"):-len(".csv")]                 # e.g. "Reaching_Definitions"
    df = pd.read_csv(csv_path)
    for alias, model_ref in MODELS.items():
        records = []
        print(f"Querying {model_ref} on {safe_bm} ({len(df)} prompts)…")
        for _, row in df.iterrows():
            prompt       = row["prompt"]
            ground_truth = row["answer"]
            # call Ollama chat API and stream
            try:
                stream = ollama.chat(
                    model=model_ref,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                )
                response = ""
                for chunk in stream:
                    response += chunk["message"]["content"]
            except Exception as e:
                response = f"ERROR: {e}"
            records.append({
                "prompt":       prompt,
                "ground_truth": ground_truth,
                "response":     response
            })
        # write out only the three columns
        result_df = pd.DataFrame(records, columns=["prompt", "ground_truth", "response"])
        out_name  = f"result_{alias}_{safe_bm}.csv"
        out_path  = os.path.join(RESULT_DIR, out_name)
        result_df.to_csv(out_path, index=False)
        print(f"✓ Wrote {out_name}")