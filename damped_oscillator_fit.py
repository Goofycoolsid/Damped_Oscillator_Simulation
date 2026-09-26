"""
damped_oscillator_fit.py

A small, self-contained exercise in fitting a physical model to noisy
experimental-style data.

Simulates a damped harmonic oscillator (e.g. a pendulum losing energy to
friction/air resistance), adds realistic measurement noise, then recovers
the underlying physical parameters (natural frequency, damping ratio) by
fitting a model back to the noisy data using non-linear least squares.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

rng = np.random.default_rng(42)

# ---- 1. "Generate" noisy experimental data -------------------------------
# True underlying physical parameters (unknown to the fitting step below, just as they would be unknown for a real pendulum)
true_amplitude = 5.0      # degrees
true_damping = 0.15       # 1/s
true_omega = 4.2          # rad/s (natural angular frequency)
true_phase = 0.3          # rad

t = np.linspace(0, 15, 150)

def damped_oscillation(t, A, gamma, omega, phi):
    """Displacement of a damped harmonic oscillator over time."""
    return A * np.exp(-gamma * t) * np.cos(omega * t + phi)

clean_signal = damped_oscillation(t, true_amplitude, true_damping, true_omega, true_phase)
noise = rng.normal(0, 0.25, size=t.shape)   # simulated sensor noise
measured = clean_signal + noise

# ---- 2. Fit the model back to the noisy data ------------------------------
# Initial guesses matters for convergence with a non-linear, oscillatory model
initial_guess = [4.0, 0.1, 4.0, 0.0]

params_fit, covariance = curve_fit(damped_oscillation, t, measured, p0=initial_guess)
A_fit, gamma_fit, omega_fit, phi_fit = params_fit
param_errors = np.sqrt(np.diag(covariance))

print("Fitted parameters (with 1-sigma uncertainty):")
print(f"  Amplitude       A     = {A_fit:.3f} +/- {param_errors[0]:.3f} deg")
print(f"  Damping coeff.  gamma = {gamma_fit:.3f} +/- {param_errors[1]:.3f} 1/s")
print(f"  Angular freq.   omega = {omega_fit:.3f} +/- {param_errors[2]:.3f} rad/s")
print(f"  Phase           phi   = {phi_fit:.3f} +/- {param_errors[3]:.3f} rad")

# ---- 3. Goodness of fit (R-squared) ---------------------------------------
residuals = measured - damped_oscillation(t, *params_fit)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((measured - np.mean(measured))**2)
r_squared = 1 - ss_res / ss_tot
print(f"\nR-squared of fit: {r_squared:.4f}")

# ---- 4. Plot ---------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]})

axes[0].scatter(t, measured, s=12, alpha=0.5, label="Noisy measurements")
axes[0].plot(t, damped_oscillation(t, *params_fit), color="crimson", lw=2, label="Fitted model")
axes[0].set_ylabel("Angular displacement (deg)")
axes[0].set_title("Damped oscillator: model fit to noisy data")
axes[0].legend()

axes[1].scatter(t, residuals, s=12, color="grey")
axes[1].axhline(0, color="black", lw=1)
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Residual")

plt.tight_layout()
plt.savefig("damped_oscillator_fit.png", dpi=150)
print("\nSaved plot to damped_oscillator_fit.png")