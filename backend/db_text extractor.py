import pandas as pd

# Load CSV
df = pd.read_csv("input.csv")

# Ensure required columns exist
if not {"description", "img_url"}.issubset(df.columns):
    raise ValueError("CSV must contain 'description' and 'img_url' columns")

# Extract content
for idx, row in df.iterrows():
    description = row["description"]
    img_url = row["img_url"]

    print(f"Row {idx+1}:")
    print(f"  Description: {description}")
    print(f"  Image URL: {img_url}")
    print("-" * 40)

    