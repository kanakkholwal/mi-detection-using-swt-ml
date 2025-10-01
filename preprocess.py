import os
import csv

DATA_DIR = "./ptbdb"   # root of your dataset

output_file = os.path.join(DATA_DIR, "PTB_LABELS.csv")

rows = []
for root, _, files in os.walk(DATA_DIR):
    for f in files:
        if f.endswith(".hea"):
            record_name = os.path.splitext(f)[0]
            record_path = os.path.relpath(os.path.join(root, record_name), DATA_DIR).replace("\\","/")

            # read header
            with open(os.path.join(root, f), "r") as hf:
                header = hf.read().lower()

            # look for diagnosis keywords
            if "inferior" in header and "infarction" in header:
                label = "IMI"
            elif "healthy" in header:
                label = "HC"

            else:
                continue
            print(record_path)

            rows.append((record_path, label))

# write CSV
with open(output_file, "w", newline="") as cf:
    writer = csv.writer(cf)
    writer.writerow(["record","label"])
    writer.writerows(rows)

print(f"Saved {len(rows)} labeled records into {output_file}")
