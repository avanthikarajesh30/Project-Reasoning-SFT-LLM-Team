"""
prep_data.py
Author: Avanthika Rajesh
Purpose: Prepare 15k random and 15k diverse subsets from AceReason-1.1-SFT dataset
for use in LLaMA-Factory fine-tuning.
"""

import os
import json
import random
from collections import defaultdict
from datasets import load_dataset

# ----------------------------- #
# CONFIGURATION
# ----------------------------- #
RANDOM_SEED = 42
NUM_SAMPLES = 15000
SAVE_DIR = "data/prepared"

# ----------------------------- #
# HELPER FUNCTIONS
# ----------------------------- #

def load_ace_dataset():
    """Stream only the first 16k samples to avoid full download."""
    from itertools import islice
    print("🔹 Streaming AceReason-1.1-SFT dataset (no full download)...")
    dataset = load_dataset("nvidia/AceReason-1.1-SFT", split="train", streaming=True)
    data = list(islice(dataset, 16000))  # grab only ~16 k examples
    print(f"✅ Sampled {len(data)} examples from stream")
    return data

def convert_format(dataset):
    """Keep only fields needed by LLaMA-Factory: input → prompt, output → response."""
    formatted = [{"input": d["input"], "output": d["output"]} for d in dataset]
    return formatted

def sample_random(data, k):
    """Randomly select k samples."""
    random.seed(RANDOM_SEED)
    subset = random.sample(data, k)
    print(f"✅ Random subset created ({len(subset)} samples)")
    return subset

def sample_diverse(data, k):
    """
    Create a simple 'diverse' subset:
    Balance roughly across (category, source, and prompt-length bucket)
    to get variety without needing external libraries.
    """
    print("🔹 Creating diverse subset (based on category, source, and prompt length)…")

    # compute prompt length quantiles
    lengths = [len(x["input"]) for x in data]
    q1, q2 = sorted(lengths)[int(0.33*len(lengths))], sorted(lengths)[int(0.66*len(lengths))]

    def bucket(x):
        n = len(x["input"])
        return "short" if n <= q1 else ("medium" if n <= q2 else "long")

    groups = defaultdict(list)
    for i, x in enumerate(data):
        cat = x.get("category", "unknown")
        src = x.get("source", "unknown")
        bkt = bucket(x)
        groups[(cat, src, bkt)].append(i)

    random.seed(RANDOM_SEED)
    per_group = max(1, k // max(1, len(groups)))
    chosen = []
    for g, idxs in groups.items():
        random.shuffle(idxs)
        chosen.extend(idxs[:per_group])

    # top-up if fewer than k
    if len(chosen) < k:
        pool = list(set(range(len(data))) - set(chosen))
        random.shuffle(pool)
        chosen += pool[:(k - len(chosen))]

    diverse_subset = [data[i] for i in chosen[:k]]
    print(f"✅ Diverse subset created ({len(diverse_subset)} samples)")
    return diverse_subset

def save_json(data, filename):
    """Save data to JSON file."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    path = os.path.join(SAVE_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved: {path} ({len(data)} samples)")

# ----------------------------- #
# MAIN SCRIPT
# ----------------------------- #

def main():
    dataset = load_ace_dataset()
    formatted = convert_format(dataset)

    random_subset = sample_random(formatted, NUM_SAMPLES)
    diverse_subset = sample_diverse(formatted, NUM_SAMPLES)

    save_json(random_subset, "random15k.json")
    save_json(diverse_subset, "diverse15k.json")

    print("\n🎉 Done! Both subsets saved inside data/prepared/")

if __name__ == "__main__":
    main()