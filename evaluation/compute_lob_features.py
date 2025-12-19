import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR

# Adjust these if needed
REAL_LOB_PATH = DATA_DIR / "lob_features_720.npy"
GEN_LOB_PATH  = DATA_DIR / "generated_lob_samples.npy"

# ---- LOAD ----
real = np.load(REAL_LOB_PATH)
gen  = np.load(GEN_LOB_PATH)

"""
Assumption (adjust if needed):
Each snapshot shape:
[ bid_prices(10), bid_sizes(10), ask_prices(10), ask_sizes(10) ]
Total = 40 features
"""

def extract_features(x):
    bid_prices = x[:, 0:10]
    bid_sizes  = x[:, 10:20]
    ask_prices = x[:, 20:30]
    ask_sizes  = x[:, 30:40]

    best_bid = bid_prices[:, 0]
    best_ask = ask_prices[:, 0]

    spread = best_ask - best_bid
    depth_bid = bid_sizes.sum(axis=1)
    depth_ask = ask_sizes.sum(axis=1)

    return spread, depth_bid, depth_ask

# ---- COMPUTE ----
spread_real, depth_bid_real, depth_ask_real = extract_features(real)
spread_gen,  depth_bid_gen,  depth_ask_gen  = extract_features(gen)

# ---- SAVE ----
np.save(OUT_DIR / "spread_real.npy", spread_real)
np.save(OUT_DIR / "spread_gen.npy", spread_gen)
np.save(OUT_DIR / "depth_bid_real.npy", depth_bid_real)
np.save(OUT_DIR / "depth_bid_gen.npy", depth_bid_gen)
np.save(OUT_DIR / "depth_ask_real.npy", depth_ask_real)
np.save(OUT_DIR / "depth_ask_gen.npy", depth_ask_gen)

print("LOB feature arrays saved successfully.")
