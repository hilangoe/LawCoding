import random

INPUT_FILE = "../data/training_data.jsonl"
OUTPUT_FILE = "../data/training_data_dev.jsonl"
N = 200        # number of rows
SEED = 42

random.seed(SEED)

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

sample = random.sample(lines, min(N, len(lines)))

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for line in sample:
        f.write(line)

print(f"Wrote {len(sample)} rows to {OUTPUT_FILE}")