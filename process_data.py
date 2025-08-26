import numpy as np

# Load data
print("Loading data...")
lob = np.load('data/lob.npy', allow_pickle=True)
print(f"Loaded {len(lob)} snapshots")

# Process all snapshots
X_list = []
for i in range(len(lob)):
    snapshot = lob[i]
    bids = snapshot[0]
    asks = snapshot[1]
    
    # Calculate mid price
    best_bid = float(bids[0, 0])
    best_ask = float(asks[0, 0])
    mid = 0.5 * (best_bid + best_ask)
    
    # Extract columns
    bid_prices = bids[:, 0].astype(float)
    ask_prices = asks[:, 0].astype(float)
    bid_volumes = bids[:, 1].astype(float)
    ask_volumes = asks[:, 1].astype(float)
    
    # Build features
    features = np.concatenate([
        (bid_prices - mid) / mid,
        (ask_prices - mid) / mid,
        np.log1p(bid_volumes),
        np.log1p(ask_volumes)
    ])
    
    X_list.append(features)
    
    if i % 100 == 0:
        print(f"Processed {i}/{len(lob)} snapshots")

# Convert to array
X = np.array(X_list)
print(f"Final feature matrix shape: {X.shape}")

# Save results
np.save('data/lob_features_720.npy', X)
print("Features saved to data/lob_features_720.npy")

# Show statistics
print("\nFeature statistics:")
print("Mean:", X.mean(axis=0))
print("Std:", X.std(axis=0))