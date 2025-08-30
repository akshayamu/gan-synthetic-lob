import pandas as pd
import numpy as np

# Number of snapshots (1 hour of 5-second intervals)
n_rows = 720

# Generate synthetic LOB data
np.random.seed(42)
bid_1 = np.random.uniform(30000, 31000, n_rows)       # bid prices
ask_1 = bid_1 + np.random.uniform(0.5, 5, n_rows)     # ask slightly above bid
bid_vol_1 = np.random.uniform(0.1, 5, n_rows)         # bid volumes
ask_vol_1 = np.random.uniform(0.1, 5, n_rows)         # ask volumes

# Create DataFrame
df = pd.DataFrame({
    'bid_1': bid_1,
    'ask_1': ask_1,
    'bid_vol_1': bid_vol_1,
    'ask_vol_1': ask_vol_1
})

# Save CSV
df.to_csv('btc_lob_5s_1h.csv', index=False)
print("CSV 'btc_lob_5s_1h.csv' created successfully!")
