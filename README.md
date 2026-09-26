# Damped Oscillator Simulation: Model Fitting to Noisy Data

A short project simulating a damped harmonic oscillator (e.g. a pendulum losing energy to friction), adding realistic measurement noise, and recovering the underlying physical parameters — amplitude, damping coefficient, natural frequency, phase — using non-linear least-squares fitting.

**Libraries:** NumPy, SciPy (`optimize.curve_fit`), Matplotlib

**Approach:** Generated synthetic time-series data from a known damped-oscillation model with added Gaussian noise, then fit the same functional form back to the noisy data to recover the original parameters and their uncertainties, and evaluated the fit with R².

**Result:** Recovered all four parameters within their 1-sigma uncertainty of the true values, with R² = 0.986.

**Challenge:** Non-linear fits to oscillatory functions are sensitive to the initial parameter guess — a poor starting guess for frequency in particular can converge to a completely wrong (aliased) solution, so getting the initial guess into a sensible range mattered more than it would for a simple linear fit.

## Files
- `damped_oscillator_fit.py` — the full simulation, fitting, and plotting code
- `damped_oscillator_fit.png` — output plot (fit vs. noisy data, plus residuals)
