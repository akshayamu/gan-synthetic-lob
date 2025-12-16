import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from evaluation.lob_metrics import compute_spread, compute_depth


def compare(real, synth):
    metrics = {}

    for name, fn in {
        "spread": compute_spread,
        "depth": compute_depth,
    }.items():
        r, s = fn(real), fn(synth)
        metrics[name] = {
            "KS": ks_2samp(r, s).statistic,
            "W1": wasserstein_distance(r, s),
            "real_mean": r.mean(),
            "synth_mean": s.mean()
        }
    return metrics
