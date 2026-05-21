import csv
from pathlib import Path

source = Path("metadata_with_jaguar.csv")
output = Path("metadata_with_jaguar_cleaned.csv")
expected_columns = ["identity", "path", "date", "orientation", "species", "dataset"]

kept = 0
removed = 0

with source.open(newline="", encoding="utf-8-sig") as src, output.open(
    "w", newline="", encoding="utf-8"
) as dst:
    reader = csv.DictReader(src)

    if reader.fieldnames != expected_columns:
        raise ValueError(f"Unexpected columns: {reader.fieldnames!r}")

    writer = csv.DictWriter(dst, fieldnames=expected_columns)
    writer.writeheader()

    for row in reader:
        if Path(row["path"]).is_file():
            writer.writerow({column: row.get(column, "") for column in expected_columns})
            kept += 1
        else:
            removed += 1

print(f"kept={kept}")
print(f"removed={removed}")
print(f"output={output}")