from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

print("========== EXISTING CSV FILES ==========")

csv_files = list(DATA_DIR.rglob("*.csv"))

for file in csv_files:
    print(f"\nFile: {file.relative_to(PROJECT_ROOT)}")

    df = pd.read_csv(file, nrows=5)

    print("Columns:")
    print(df.columns.tolist())

    print("\nSample:")
    print(df.head(2).to_string(index=False))