from __future__ import annotations

from pathlib import Path

import numpy as np

from ece132a import calc_error_rate, get_const, load_signal, min_dist_detection

N = 1000
SAMPLES_PER_SYMBOL = 10
NUM_PILOTS = min(400, N // 2)
EQ_TAPS = 7

# Sweep grids (edit these lists as needed)
ALPHAS = [0.01, 0.02, 0.03, 0.05]
MU_TRAINS = [0.001, 0.002, 0.003]
MU_DDS = [0.0002, 0.0005, 0.0008]

LAB3_DIR = Path(__file__).resolve().parent / "lab3"
z_path = LAB3_DIR / "zadoff_chu_sequence2.npy"
mf_out_path = LAB3_DIR / "mf_output2.npy"
tx_payload_path = LAB3_DIR / "tx_payload_symbols2.npy"

for p in (z_path, mf_out_path, tx_payload_path):
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run lab31->sdr_lab3->lab33->lab34 first.")

z = load_signal(str(z_path))
rx_signal = load_signal(str(mf_out_path))
const = get_const("64-QAM", 1)
tx_symbols = load_signal(str(tx_payload_path)).ravel()
pilot_tx = tx_symbols[:NUM_PILOTS]

payload_start = (len(z) + int(N / 10)) * SAMPLES_PER_SYMBOL

# One-time timing/CFO/channel coarse recovery from pilots.
best_mse = np.inf
best_omega = 0.0
best_h = 1.0 + 0j
best_rx_symbols = None
for phase in range(SAMPLES_PER_SYMBOL):
    cand = rx_signal[
        payload_start + phase : payload_start + phase + N * SAMPLES_PER_SYMBOL : SAMPLES_PER_SYMBOL
    ]
    if len(cand) != N:
        continue

    pilot_rx = cand[:NUM_PILOTS]
    ratio = pilot_rx / pilot_tx
    ang = np.unwrap(np.angle(ratio))
    n = np.arange(NUM_PILOTS)
    omega, _phi = np.polyfit(n, ang, 1)

    n_all = np.arange(N)
    cand_derot = cand * np.exp(-1j * omega * n_all)
    pilot_derot = cand_derot[:NUM_PILOTS]

    den = np.vdot(pilot_tx, pilot_tx)
    if np.abs(den) < 1e-12:
        continue

    h = np.vdot(pilot_tx, pilot_derot) / den
    err = pilot_derot - h * pilot_tx
    mse = np.mean(np.abs(err) ** 2)

    if mse < best_mse:
        best_mse = mse
        best_omega = omega
        best_h = h
        best_rx_symbols = cand

if best_rx_symbols is None:
    raise RuntimeError("failed to recover symbols: no valid timing phase candidates")

n_all = np.arange(N)
rx_base = best_rx_symbols * np.exp(-1j * best_omega * n_all) / best_h

best = None
for alpha in ALPHAS:
    # Residual phase tracking
    rx_eq = rx_base.copy()
    rx_hard = min_dist_detection(rx_eq, const).ravel()
    phase_err = np.angle(rx_eq * np.conj(rx_hard))
    theta = np.zeros_like(phase_err)
    for k in range(1, len(theta)):
        theta[k] = theta[k - 1] + alpha * phase_err[k]
    rx_eq = rx_eq * np.exp(-1j * theta)

    for mu_train in MU_TRAINS:
        for mu_dd in MU_DDS:
            # LMS block
            half = EQ_TAPS // 2
            w = np.zeros(EQ_TAPS, dtype=np.complex128)
            w[half] = 1.0 + 0j
            rx_pad = np.pad(rx_eq, (half, half), mode="constant")
            rx_lms = np.zeros_like(rx_eq, dtype=np.complex128)

            for n in range(len(rx_eq)):
                x = rx_pad[n : n + EQ_TAPS][::-1]
                y = np.vdot(w, x)
                if n < NUM_PILOTS:
                    d = tx_symbols[n]
                    e = d - y
                    w += mu_train * np.conj(e) * x
                else:
                    d = min_dist_detection(np.array([y]), const).ravel()[0]
                    e = d - y
                    w += mu_dd * np.conj(e) * x
                rx_lms[n] = y

            rx_detected = min_dist_detection(rx_lms, const)
            ser = calc_error_rate(tx_symbols, rx_detected)

            print(
                f"SER={ser:.6f} alpha={alpha:.4f} "
                f"mu_train={mu_train:.6f} mu_dd={mu_dd:.6f}"
            )

            if best is None or ser < best["ser"]:
                best = {
                    "ser": ser,
                    "alpha": alpha,
                    "mu_train": mu_train,
                    "mu_dd": mu_dd,
                }

if best is not None:
    print(
        "BEST "
        f"SER={best['ser']:.6f} alpha={best['alpha']:.4f} "
        f"mu_train={best['mu_train']:.6f} mu_dd={best['mu_dd']:.6f}"
    )
