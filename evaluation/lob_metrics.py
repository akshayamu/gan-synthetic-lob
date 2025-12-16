import numpy as np

def compute_spread(X):
    # X shape: (N, 40)
    best_bid = X[:, 9]   # last bid price feature
    best_ask = X[:, 10]  # first ask price feature
    return best_ask - best_bid

def compute_depth(X):
    bid_depth = np.sum(np.expm1(X[:, 20:30]), axis=1)
    ask_depth = np.sum(np.expm1(X[:, 30:40]), axis=1)
    return bid_depth + ask_depth
