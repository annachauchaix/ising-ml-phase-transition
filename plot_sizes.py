"""
Compare several lattice sizes on one plot.
First run ising_mc.py for each L (change L at the top of that file).
"""

import matplotlib.pyplot as plt
import numpy as np

sizes = [8, 16, 32]      # the L you have already calculated

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

for L in sizes:
    # columns: T, |m|, err, e, err, C, err, chi, err
    data = np.loadtxt(f"results_L{L}.dat")
    T = data[:, 0]
    C, C_err = data[:, 5], data[:, 6]
    chi, chi_err = data[:, 7], data[:, 8]

    ax[0].errorbar(T, C, yerr=C_err, fmt="o-", ms=3, capsize=2, label=f"L = {L}")
    ax[1].errorbar(T, chi, yerr=chi_err, fmt="o-", ms=3, capsize=2, label=f"L = {L}")

ax[0].set_ylabel("Heat capacity C")
ax[1].set_ylabel("Susceptibility chi")
for a in ax:
    a.axvline(2.269, color="grey", ls="--", label="Tc exact = 2.269")
    a.set_xlabel("T")
    a.legend()

fig.tight_layout()
fig.savefig("ising_sizes.png", dpi=120)
plt.show()
