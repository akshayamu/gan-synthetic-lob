import numpy as np
from scipy import stats

# Load real and synthetic data
print("Loading data...")
real_data = np.load('data/lob_features_720.npy')
synthetic_data = np.load('data/generated_lob_samples.npy')

print(f"Real data shape: {real_data.shape}")
print(f"Synthetic data shape: {synthetic_data.shape}")

# Perform KS test on each feature
ks_statistics = []
print("\nPerforming KS-tests...")
for i in range(min(real_data.shape[1], synthetic_data.shape[1])):
    ks_stat, p_value = stats.ks_2samp(real_data[:, i], synthetic_data[:, i])
    ks_statistics.append(ks_stat)

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
    print("❌ Mean KS-statistic >= 0.05")

print(f"\n🎯 Your Result: {mean_ks:.4f}")
print("Success criterion: < 0.05")

# Show first few KS statistics
print("\nFirst 10 feature KS stats:")
for i, ks in enumerate(ks_statistics[:10]):
    status = "✅" if ks < 0.05 else "❌"
    print(f"  Feature {i}: {ks:.4f} {status}")