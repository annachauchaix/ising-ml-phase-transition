"""
ML: logistic regression on one number, |m|.

The model learns "ordered or disordered" only from snapshots far from Tc.
Then it predicts for all temperatures. Where the probability of
"ordered" is 50%, that is Tc according to the model.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

TC_EXACT = 2.269
L = 16

# ---------------- 1. Load the data ----------------
data = np.load(f"snapshots_L{L}.npz")
spins = data["spins"]          # shape (number of snapshots, L, L)
T = data["T"]                  # temperature of each snapshot

# one feature per snapshot: |m|
abs_m = np.abs(spins.mean(axis=(1, 2)))
X = abs_m.reshape(-1, 1)       

# label: 1 = ordered (cold), 0 = disordered (hot)
y = (T < TC_EXACT).astype(int)

# ---------------- 2. Train only far from Tc ----------------
far = (T < 2.0) | (T > 2.6)    
X_far, y_far = X[far], y[far]

X_train, X_test, y_train, y_test = train_test_split(
    X_far, y_far, test_size=0.2, random_state=42, stratify=y_far)

model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Train: {len(y_train)} snapshots, test: {len(y_test)} snapshots")
print(f"Accuracy on the test set: {accuracy:.3f}")

# ---------------- 3. Ask the model about ALL temperatures ----------------
p_ordered = model.predict_proba(X)[:, 1]     # probability of "ordered"

temps = np.unique(T)                         # each temperature once, sorted
p_mean = np.array([p_ordered[T == t].mean() for t in temps])

# Tc = where the mean probability crosses 0.5 (linear interpolation)
# p goes from 1 (cold) to 0 (hot); np.interp needs increasing x, so we flip
Tc_model = np.interp(0.5, p_mean[::-1], temps[::-1])
print(f"\nTc from the model (p = 0.5): {Tc_model:.3f}")
print(f"Tc exact (Onsager):          {TC_EXACT}")

# ---------------- 4. Plot ----------------
plt.figure(figsize=(7, 4.5))
plt.axvspan(2.0, 2.6, color="grey", alpha=0.15, label="not used for training")
plt.plot(temps, p_mean, "o-", label="model: probability of 'ordered'")
plt.axhline(0.5, color="grey", lw=1)
plt.axvline(TC_EXACT, color="black", ls="--", label=f"Tc exact = {TC_EXACT}")
plt.axvline(Tc_model, color="tab:red", ls=":", label=f"Tc model = {Tc_model:.3f}")
plt.xlabel("T")
plt.ylabel("Probability of ordered phase")
plt.title(f"Logistic regression on |m|, L = {L}")
plt.legend()
plt.tight_layout()
plt.savefig(f"ml_level1_L{L}.png", dpi=120)
plt.show()
