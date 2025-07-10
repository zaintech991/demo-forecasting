import csv

# Read unique grocery items from file
with open("scripts/grocery_items_865.txt", "r", encoding="utf-8") as f:
    GROCERY_ITEMS = [line.strip() for line in f if line.strip()]

with open("data_export/product_hierarchy.csv", newline="") as infile, open(
    "data_export/product_hierarchy_updated.csv", "w", newline=""
) as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    header = next(reader)
    writer.writerow(header)

    for idx, row in enumerate(reader):
        if idx < len(GROCERY_ITEMS):
            row[5] = GROCERY_ITEMS[idx]  # Assign unique name in order
        writer.writerow(row)

print(
    f"Updated product names written to data_export/product_hierarchy_updated.csv using {len(GROCERY_ITEMS)} unique grocery items."
)
