import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from lob_metrics import compute_depth

real = np.load("data/lob_features_720.npy")
synthetic = np.load("data/generated_lob_samples.npy")

depth = compute_depth(real)
y = (depth > np.median(depth)).astype(int)

Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    real, y, test_size=0.3, random_state=42
)

clf_real = LogisticRegression(max_iter=2000).fit(Xr_tr, yr_tr)
acc_real = accuracy_score(yr_te, clf_real.predict(Xr_te))

X_aug = np.vstack([Xr_tr, synthetic])
y_aug = np.concatenate([yr_tr, yr_tr[:len(synthetic)]])

clf_aug = LogisticRegression(max_iter=2000).fit(X_aug, y_aug)
acc_aug = accuracy_score(yr_te, clf_aug.predict(Xr_te))

print(f"Real only accuracy: {acc_real:.3f}")
print(f"Real + synthetic accuracy: {acc_aug:.3f}")
