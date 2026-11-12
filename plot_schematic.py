"""
Schematic representation of a configuration of the 2D Ising model on a square lattice:
spins (arrows up / down) on the sites of the lattice.

Saves figures/ising_schematic.png
"""

import matplotlib.pyplot as plt
import numpy as np

L = 6                                   # small lattice, so every arrow is visible
UP, DOWN = "#d9493b", "#2a6fdb"         # spin +1 = red, spin -1 = blue
GRID = "#c8c8c8"

rng = np.random.default_rng(3)
spins = rng.choice([-1, 1], size=(L, L))

fig, ax = plt.subplots(figsize=(5.5, 6.2))

# lattice bonds
for k in range(L):
    ax.plot([0, L - 1], [k, k], color=GRID, lw=1.2, zorder=1)
    ax.plot([k, k], [0, L - 1], color=GRID, lw=1.2, zorder=1)

# spins as arrows
for i in range(L):
    for j in range(L):
        s = spins[i, j]
        ax.annotate("", xy=(j, i + 0.32 * s), xytext=(j, i - 0.32 * s),
                    arrowprops=dict(arrowstyle="-|>", color=UP if s > 0 else DOWN,
                                    lw=2.2, mutation_scale=16), zorder=3)

ax.set_xlim(-0.7, L - 0.3)
ax.set_ylim(-0.7, L - 0.3)
ax.set_aspect("equal")
ax.axis("off")

fig.text(0.5, 0.04,
         "Schematic representation of a configuration\nof the 2D Ising model on a square lattice.",
         ha="center", va="bottom", fontsize=11.5)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig("figures/ising_schematic.png", dpi=150, facecolor="white")
plt.show()
