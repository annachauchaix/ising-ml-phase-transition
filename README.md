# Machine learning finds the phase transition in the 2D Ising model

A physicist's bridge project: Monte Carlo simulation (my PhD toolbox) + machine learning (scikit-learn) applied to the same problem. The question: Сan a model find the critical temperature without being told any physics?

![Monte Carlo results](figures/ising_L16.png)

## The physics

The 2D Ising model is the simplest model of a magnet: spins ±1 on an L×L square lattice with periodic boundaries, H = −J Σ sᵢsⱼ. Below the critical temperature the spins order (ferromagnet), above it thermal noise wins (paramagnet). The exact answer is known (Onsager 1944): Tc = 2.269 (J = 1, k_B = 1), which makes this model a perfect benchmark.

## What is in the project

**1. Monte Carlo simulation** (`ising_mc.py`) — Metropolis algorithm, temperature scan from hot to cold, 10 independent runs with different seeds for error bars. Observables: |m|, energy, heat capacity C, susceptibility χ. Tc is located from the peaks of C and χ.

**2. Finite-size check** (`plot_sizes.py`) — the same scan for L = 8, 16, 32: the peaks grow and move toward the exact Tc as L grows, as expected for a true phase transition.

**3. Machine learning** (`ml_level1.py`) — logistic regression trained on lattice snapshots labelled "ordered / disordered". The model is trained **only far from the transition** (T < 2.0 and T > 2.6), then asked about all temperatures. The temperature where its predicted probability crosses 50% is the model's estimate of Tc — it reconstructs the transition in a region it has never seen.

![ML result](figures/ml_level1_L16.png)

## Results

| Method | Tc (L = 16) | Tc (L = 32) |
|---|---|---|
| Peak of heat capacity C | 2.292 ± 0.016 | 2.270 ± 0.011 |
| Peak of susceptibility χ | 2.346 ± 0.021 | 2.304 ± 0.013 |
| Logistic regression (p = 0.5) | 2.31 | — |
| **Exact (Onsager)** | **2.269** (infinite lattice) | |

On a finite lattice the transition is smeared and the apparent Tc is shifted
upward — the results agree with the exact value within finite-size effects.

## How to run

```bash
pip install -r requirements.txt
python ising_mc.py        # ~15 min for L = 16: scan + snapshots_L16.npz
python ml_level1.py       # trains the model on the snapshots, ~1 s
python plot_sizes.py      # after running ising_mc.py for L = 8, 16, 32
```

All random seeds are fixed, so the numbers above are exactly reproducible.

## Validation and honest limitations

- The simulation is checked against exact results: e(T = 1.5) = −1.950 vs
  Onsager's −1.951; |m|(T) follows Yang's exact curve.
- Snapshots 20 sweeps apart are correlated near Tc (autocorrelation time
  ~40 sweeps there), so the test accuracy is slightly optimistic.
- The ML estimate of Tc depends mildly on the training window (±0.03); it is
  a demonstration of the method, not a precision measurement.

## Roadmap

- [ ] Logistic regression on raw spins: the Z2 symmetry trap (77% → 99%)
- [ ] Unsupervised: PCA rediscovers the order parameter
- [ ] Finite-size scaling of the ML estimate of Tc

## References

- L. Onsager, Phys. Rev. 65, 117 (1944) — exact solution
- J. Carrasquilla & R. Melko, *Machine learning phases of matter*,
  Nature Physics 13, 431 (2017) — the idea of this project
