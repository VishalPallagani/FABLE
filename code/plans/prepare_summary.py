import os
import json

def analyze_plan_steps(directory):
    min_steps = float('inf')
    max_steps = float('-inf')
    total_steps = 0
    total_files = 0

    min_file = None
    max_file = None

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            with open(file_path, "r") as file:
                data = json.load(file)
                plan_steps = data.get("plan_steps")

                # Only proceed if plan_steps exists and is non-empty
                if plan_steps:
                    steps_count = len(plan_steps)
                    
                    if steps_count < min_steps:
                        min_steps = steps_count
                        min_file = filename
                    
                    if steps_count > max_steps:
                        max_steps = steps_count
                        max_file = filename
                    
                    total_steps += steps_count
                    total_files += 1

    if total_files == 0:
        print("No valid plans found.")
        return

    avg_steps = total_steps / total_files

    print(f"Minimum plan steps: {min_steps} (File: {min_file})")
    print(f"Maximum plan steps: {max_steps} (File: {max_file})")
    print(f"Average plan steps: {avg_steps:.2f}")

directory_path = "../raw_data/planning/"  # replace this
analyze_plan_steps(directory_path)
