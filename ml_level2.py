"""
ML level 2: logistic regression on the raw lattice (all L*L spins).

No |m| this time: the model sees every spin.
Three experiments:
  A. raw spins                       -> looks perfect...
  B. same test, all spins flipped    -> ...but the model fails (a trap!)
  C. flipped copies added to data    -> honest accuracy drops
  D. fix: align the sign of each lattice -> works again, find Tc
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

TC_EXACT = 2.269
L = 16

# ---------------- Load the data ----------------
data = np.load(f"snapshots_L{L}.npz")
spins = data["spins"]                      # shape (4900, L, L)
T = data["T"]

X = spins.reshape(len(spins), -1)          # each lattice -> one row of L*L numbers
y = (T < TC_EXACT).astype(int)             # 1 = ordered, 0 = disordered
far = (T < 2.0) | (T > 2.6)                # train only far from Tc


def train_and_test(X, y):
    """Split into train/test, train logistic regression, return model and test data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    # no class_weight="balanced" here (unlike ml_level1.py): this script is about the
    # symmetry trap, and without reweighting the "learned nothing" baseline in part C is clearer
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)
    return model, X_test, y_test


# ---------------- A. Raw spins ----------------
model_raw, X_test, y_test = train_and_test(X[far], y[far])
print(f"A. Raw spins:                     accuracy = {model_raw.score(X_test, y_test):.3f}")

# ---------------- B. The trap ----------------
# Flip every spin of the cold test lattices: physically the SAME ordered state
cold = y_test == 1
acc_cold = model_raw.score(X_test[cold], y_test[cold])
acc_cold_flipped = model_raw.score(-X_test[cold], y_test[cold])
print(f"B. Cold lattices:                 accuracy = {acc_cold:.3f}")
print(f"   Same cold lattices, flipped:   accuracy = {acc_cold_flipped:.3f}")

# ---------------- C. Honest data: both signs ----------------
X_both = np.vstack([X[far], -X[far]])      # every lattice + its flipped copy
y_both = np.concatenate([y[far], y[far]])
model_both, X_test, y_test = train_and_test(X_both, y_both)
baseline = max(y_test.mean(), 1 - y_test.mean())   # always answer the most common class
print(f"C. Both signs in the data:        accuracy = {model_both.score(X_test, y_test):.3f}")
print(f"   (always answering 'disordered' would give {baseline:.3f} -> the model learned nothing)")

# ---------------- D. Fix: align the sign ----------------
# multiply each lattice by the sign of its magnetization -> m is always >= 0
sign = np.sign(X.sum(axis=1))
sign[sign == 0] = 1
X_aligned = X * sign[:, None]

model_al, X_test, y_test = train_and_test(X_aligned[far], y[far])
print(f"D. Aligned sign:                  accuracy = {model_al.score(X_test, y_test):.3f}")

# Tc from the aligned model: where the mean probability of "ordered" crosses 0.5
p_ordered = model_al.predict_proba(X_aligned)[:, 1]
temps = np.unique(T)
p_mean = np.array([p_ordered[T == t].mean() for t in temps])
for k in range(len(temps) - 1):
    if p_mean[k] >= 0.5 > p_mean[k + 1]:
        Tc_model = temps[k] + (p_mean[k] - 0.5) / (p_mean[k] - p_mean[k + 1]) * (temps[k + 1] - temps[k])
        break
print(f"\nTc from the model (p = 0.5): {Tc_model:.3f}")
print(f"Tc exact (Onsager):          {TC_EXACT}")

# ---------------- Plots ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

# left: what the raw model learned, one weight per spin
w = model_raw.coef_[0].reshape(L, L)
im = ax[0].imshow(w, cmap="RdBu_r", vmin=-np.abs(w).max(), vmax=np.abs(w).max())
fig.colorbar(im, ax=ax[0])
ax[0].set_title("Model A: weight of each spin\n(almost all positive = it learned 'sum of spins')")
ax[0].set_xticks([])
ax[0].set_yticks([])

# right: probability of "ordered" for the aligned model
ax[1].axvspan(2.0, 2.6, color="grey", alpha=0.15, label="not used for training")
ax[1].plot(temps, p_mean, "o-", label="model D: probability of 'ordered'")
ax[1].axhline(0.5, color="grey", lw=1)
ax[1].axvline(TC_EXACT, color="black", ls="--", label=f"Tc exact = {TC_EXACT}")
ax[1].axvline(Tc_model, color="tab:red", ls=":", label=f"Tc model = {Tc_model:.3f}")
ax[1].set_xlabel("T")
ax[1].set_ylabel("Probability of ordered phase")
ax[1].set_title(f"Raw spins, sign aligned, L = {L}")
ax[1].legend()

fig.tight_layout()
fig.savefig(f"figures/ml_level2_L{L}.png", dpi=120)
plt.show()
