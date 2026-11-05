"""
Monte Carlo (Metropolis) for the 2D Ising model on an L x L square lattice,
with several independent runs to get mean values and error bars.

H = -J * sum over nearest neighbours of s_i * s_j,  J = 1,  k_B = 1
Periodic boundary conditions.

Each run = full temperature scan with its own random seed.
Result = mean over runs, error = standard deviation / sqrt(number of runs).
Exact answer (Onsager): Tc = 2.269
"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np

# ---------------- Parameters ----------------
L = 16                    # lattice size
N = L * L                 # number of spins
N_THERM = 1000            # sweeps for thermalization
N_MEAS = 2000             # sweeps for measurements
N_RUNS = 10               # number of independent runs
SNAPSHOT_EVERY = 20       # save a copy of the lattice every 20 sweeps

# coarse far from Tc, fine near Tc (from hot to cold)
coarse_hot = np.linspace(3.5, 2.7, 9)      # step 0.1
fine = np.linspace(2.68, 2.0, 35)          # step 0.02
coarse_cold = np.linspace(1.9, 1.5, 5)     # step 0.1
temperatures = np.concatenate([coarse_hot, fine, coarse_cold])


# ---------------- Functions ----------------
def total_energy(s):
    """Energy of the whole lattice. Each bond counted once (right + down)."""
    E = 0
    for i in range(L):
        for j in range(L):
            E -= s[i][j] * (s[(i + 1) % L][j] + s[i][(j + 1) % L])
    return E


def sweep(s, T, E, M):
    """One sweep = N attempts to flip a random spin. Returns new E and M."""
    for _ in range(N):
        i = random.randrange(L)
        j = random.randrange(L)
        neighbours = (s[(i + 1) % L][j] + s[(i - 1) % L][j]
                      + s[i][(j + 1) % L] + s[i][(j - 1) % L])
        dE = 2 * s[i][j] * neighbours
        if dE <= 0 or random.random() < math.exp(-dE / T):
            s[i][j] = -s[i][j]
            E += dE
            M += 2 * s[i][j]
    return E, M


def one_run(seed):
    """One full temperature scan. Returns a table: rows = T, columns = |m|, e, C, chi."""
    random.seed(seed)
    s = [[random.choice([-1, 1]) for _ in range(L)] for _ in range(L)]
    E = total_energy(s)
    M = sum(sum(row) for row in s)

    table = []
    snapshots = []      # copies of the lattice
    snapshot_T = []     # temperature of each copy
    for T in temperatures:
        # thermalization
        for _ in range(N_THERM):
            E, M = sweep(s, T, E, M)

        # measurements
        E_sum = E2_sum = M_sum = M2_sum = 0.0
        for step in range(N_MEAS):
            E, M = sweep(s, T, E, M)
            E_sum += E
            E2_sum += E * E
            M_sum += abs(M)
            M2_sum += M * M
            if step % SNAPSHOT_EVERY == 0:
                snapshots.append([row[:] for row in s])
                snapshot_T.append(T)

        E_av = E_sum / N_MEAS
        E2_av = E2_sum / N_MEAS
        M_av = M_sum / N_MEAS
        M2_av = M2_sum / N_MEAS

        C = (E2_av - E_av ** 2) / (T * T * N)      # heat capacity per spin
        chi = (M2_av - M_av ** 2) / (T * N)        # susceptibility per spin
        table.append([M_av / N, E_av / N, C, chi])
    return np.array(table), snapshots, snapshot_T


# ---------------- Main program: several runs ----------------
all_runs = []
for run in range(N_RUNS):
    print(f"Run {run + 1} of {N_RUNS} ...")
    table, snapshots, snapshot_T = one_run(seed=run)
    all_runs.append(table)
    if run == 0:   # snapshots for ML: from the first run only
        np.savez_compressed(f"snapshots_L{L}.npz",
                            spins=np.array(snapshots, dtype=np.int8),
                            T=np.array(snapshot_T))

all_runs = np.array(all_runs)        # shape: (N_RUNS, number of T, 4)

mean = all_runs.mean(axis=0)                             # average over runs
error = all_runs.std(axis=0) / np.sqrt(N_RUNS)           # error of the mean

# ---------------- Tc from the peaks ----------------
T_all = temperatures
# position of the peak in each run separately -> Tc with an error bar
Tc_C_runs = T_all[np.argmax(all_runs[:, :, 2], axis=1)]
Tc_chi_runs = T_all[np.argmax(all_runs[:, :, 3], axis=1)]
print(f"\nL = {L}, {N_RUNS} runs")
print(f"Tc from peak of C:   {Tc_C_runs.mean():.3f} +- {Tc_C_runs.std() / np.sqrt(N_RUNS):.3f}")
print(f"Tc from peak of chi: {Tc_chi_runs.mean():.3f} +- {Tc_chi_runs.std() / np.sqrt(N_RUNS):.3f}")
print("Tc exact (Onsager):  2.269")

# ---------------- Save ----------------
# columns: T, |m|, err, e, err, C, err, chi, err
order = np.argsort(T_all)                       # from cold to hot
columns = [T_all]
for k in range(4):
    columns += [mean[:, k], error[:, k]]
table = np.column_stack(columns)[order]
np.savetxt(f"results_L{L}.dat", table, fmt="%10.5f",
           header="T  |m| err  e err  C err  chi err")

# ---------------- Plots ----------------
names = ["|m| per spin", "Energy per spin", "Heat capacity C", "Susceptibility chi"]
fig, ax = plt.subplots(2, 2, figsize=(10, 7))
for k, a in enumerate(ax.flat):
    a.errorbar(T_all[order], mean[order, k], yerr=error[order, k],
               fmt="o-", ms=3, capsize=2)
    a.axvline(2.269, color="grey", ls="--", label="Tc exact = 2.269")
    a.set_xlabel("T")
    a.set_ylabel(names[k])
    a.legend()
fig.suptitle(f"2D Ising model, L = {L}, mean of {N_RUNS} runs")
fig.tight_layout()
fig.savefig(f"ising_L{L}.png", dpi=120)
plt.show()
