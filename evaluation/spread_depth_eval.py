import numpy as np
from scipy.stats import ks_2samp
import json

def summarize(x):
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95))
    }

def ks(real, gen):
    return float(ks_2samp(real, gen).statistic)

# LOAD your arrays (adjust filenames if needed)
spread_real = np.load("data/spread_real.npy")
spread_gen = np.load("data/spread_gen.npy")

depth_bid_real = np.load("data/depth_bid_real.npy")
depth_bid_gen = np.load("data/depth_bid_gen.npy")

depth_ask_real = np.load("data/depth_ask_real.npy")
depth_ask_gen = np.load("data/depth_ask_gen.npy")

metrics = {
    "spread": {
        "real": summarize(spread_real),
        "gen": summarize(spread_gen),
        "ks": ks(spread_real, spread_gen),
    },
    "depth_bid": {
        "real": summarize(depth_bid_real),
        "gen": summarize(depth_bid_gen),
        "ks": ks(depth_bid_real, depth_bid_gen),
    },
    "depth_ask": {
        "real": summarize(depth_ask_real),
        "gen": summarize(depth_ask_gen),
        "ks": ks(depth_ask_real, depth_ask_gen),
    },
}

with open("evaluation/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("Saved metrics to evaluation/metrics.json")
