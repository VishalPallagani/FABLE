import ast
import pandas as pd

df = pd.read_csv("output_data/travel_routes.csv")

# Move place names to their own columns
df[['start_name', 'destination_name']] = df['Goal'].str.split(' to ', expand=True)

# Move coordinates to their own columns
df['Domain'] = df['Domain'].apply(lambda x: ast.literal_eval(x.replace('coordinates = ', '').strip()))
df['start_coordinates'] = df['Domain'].apply(lambda x: x[0])
df['destination_coordinates'] = df['Domain'].apply(lambda x: x[1])

# Get each plan length
df["Plan"] = df["Plan"].apply(eval)
df['plan_length_1'] = df['Plan'].apply(lambda x: len(x[0]))
df['plan_length_2'] = df['Plan'].apply(lambda x: len(x[1]))
df['plan_length_3'] = df['Plan'].apply(lambda x: len(x[2]))

# Calculate average number of steps per row
def get_average_steps(row):
    return (row['plan_length_1'] + row['plan_length_2'] + row['plan_length_3']) / 3

df["average_plan_length"] = df.apply(get_average_steps, axis=1)

df[['plan_1', 'plan_2', 'plan_3']] = df['Plan'].apply(
    lambda x: pd.Series([x[0], x[1], x[2]])
)

# Drop old columns
df = df.drop(columns=[
    'Plan',
    'Goal',
    'Domain',
])

# Save large summary
df.to_csv("output_data/summary/summary_large.csv", index=False)

# Create small summary
df = df.drop(columns=[
    'plan_1',
    'plan_2',
    'plan_3',
])

df.to_csv("output_data/summary/summary_small.csv", index=False)

#
# Create overall summary card
#

# Compute the average number of steps
avg_steps = df["average_plan_length"].mean()

def max_steps_in_route(row):
    return max(row['plan_length_1'], row['plan_length_2'], row['plan_length_3'])

def min_steps_in_route(row):
    return min(row['plan_length_1'], row['plan_length_2'], row['plan_length_3'])

min_steps = df.apply(min_steps_in_route, axis=1).min()
max_steps = df.apply(max_steps_in_route, axis=1).max()

summary = f"""
# 📊 Dataset Summary

- **Total Routes:** {len(df)}
- **Average Steps per Route:** {avg_steps:.2f}
- **Min Steps in a Route:** {min_steps}
- **Max Steps in a Route:** {max_steps}

# Other Notes

This dataset contains unique routes between locations in the mainland US.
All routes are less than 150km long and within a single state.
"""

# Save to a Markdown file
with open("output_data/summary/summary.md", "w") as f:
    f.write(summary)

