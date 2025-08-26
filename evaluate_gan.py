import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load data
print("Loading data...")
real_data = np.load('data/lob_features_720.npy')
synthetic_data = np.load('data/improved_generated_lob_samples.npy')

print(f"Real data shape: {real_data.shape}")
print(f"Synthetic data shape: {synthetic_data.shape}")

# Perform KS test on each feature
ks_statistics = []
p_values = []

print("\nPerforming KS-tests on all 40 features...")
for i in range(real_data.shape[1]):
    ks_stat, p_value = stats.ks_2samp(real_data[:, i], synthetic_data[:, i])
    ks_statistics.append(ks_stat)
    p_values.append(p_value)

# Overall statistics
mean_ks = np.mean(ks_statistics)
max_ks = np.max(ks_statistics)
min_ks = np.min(ks_statistics)

print(f"\n=== KS-Test Results ===")
print(f"Mean KS-statistic: {mean_ks:.4f}")
print(f"Max KS-statistic: {max_ks:.4f}")
print(f"Min KS-statistic: {min_ks:.4f}")
print(f"Features with KS < 0.05: {np.sum(np.array(ks_statistics) < 0.05)}/{len(ks_statistics)}")

# Your success criteria
if mean_ks < 0.05:
    print("✅ SUCCESS: Mean KS-statistic < 0.05!")
else:
    print("❌ Need improvement: Mean KS-statistic >= 0.05")

print("\nIndividual KS statistics (first 10):")
for i, ks in enumerate(ks_statistics[:10]):  # Show first 10
    print(f"Feature {i}: {ks:.4f}")