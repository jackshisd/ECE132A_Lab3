from ece132a import *
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

N = 1000

symbols_per_sec = 10**5
T = 1/symbols_per_sec

samples_per_symbol = 10
LAB3_DIR = Path(__file__).resolve().parent / "lab3"
z_path = LAB3_DIR / "zadoff_chu_sequence2.npy"
mf_out_path = LAB3_DIR / "mf_output2.npy"
tx_payload_path = LAB3_DIR / "tx_payload_symbols2.npy"
if not z_path.exists():
    raise FileNotFoundError(f"Missing {z_path}. Run `python3.10 lab31.py` first.")
if not mf_out_path.exists():
    raise FileNotFoundError(
        f"Missing {mf_out_path}. Run `python3.10 lab34.py` first."
    )
if not tx_payload_path.exists():
    raise FileNotFoundError(
        f"Missing {tx_payload_path}. Run `python3.10 lab31.py` first."
    )

z = load_signal(str(z_path))
L = int((len(z) + (N+N*2/10))*samples_per_symbol)

rx_signal = load_signal(str(mf_out_path)) # loads matched-filter output

payload_start = (len(z) + int(N/10)) * samples_per_symbol

const = get_const('64-QAM', 1)
tx_symbols = load_signal(str(tx_payload_path)).ravel()

num_pilots = min(400, N // 2)
pilot_tx = tx_symbols[:num_pilots]

# Joint timing/CFO/channel estimation from pilots.
best_phase = 0
best_mse = np.inf
best_omega = 0.0
best_h = 1.0 + 0j
best_rx_symbols = None

for phase in range(samples_per_symbol):
    cand = rx_signal[payload_start + phase:payload_start + phase + N * samples_per_symbol:samples_per_symbol]
    if len(cand) != N:
        continue

    pilot_rx = cand[:num_pilots]
    ratio = pilot_rx / pilot_tx
    ang = np.unwrap(np.angle(ratio))
    n = np.arange(num_pilots)

    # Model ratio phase as linear drift: angle(ratio) ~= omega*n + phi.
    omega, _phi = np.polyfit(n, ang, 1)

    n_all = np.arange(N)
    cand_derot = cand * np.exp(-1j * omega * n_all)
    pilot_derot = cand_derot[:num_pilots]

    den = np.vdot(pilot_tx, pilot_tx)
    if np.abs(den) < 1e-12:
        continue
    h = np.vdot(pilot_tx, pilot_derot) / den
    err = pilot_derot - h * pilot_tx
    mse = np.mean(np.abs(err) ** 2)

    if mse < best_mse:
        best_mse = mse
        best_phase = phase
        best_omega = omega
        best_h = h
        best_rx_symbols = cand

if best_rx_symbols is None:
    raise RuntimeError("failed to recover symbols: no valid timing phase candidates")

n_all = np.arange(N)
rx_eq = best_rx_symbols * np.exp(-1j * best_omega * n_all) / best_h

# Decision-directed phase tracking for residual phase noise/CFO.
rx_hard = min_dist_detection(rx_eq, const).ravel()
phase_err = np.angle(rx_eq * np.conj(rx_hard))
theta = np.zeros_like(phase_err)
alpha = 0.01
for k in range(1, len(theta)):
    theta[k] = theta[k - 1] + alpha * phase_err[k]
rx_eq = rx_eq * np.exp(-1j * theta)

# 7-tap LMS equalizer: pilot-trained, then decision-directed.
eq_taps = 7
mu_train = 0.001
mu_dd = 0.0008
half = eq_taps // 2
w = np.zeros(eq_taps, dtype=np.complex128)
w[half] = 1.0 + 0j
rx_pad = np.pad(rx_eq, (half, half), mode='constant')
rx_lms = np.zeros_like(rx_eq, dtype=np.complex128)

for n in range(len(rx_eq)):
    x = rx_pad[n:n + eq_taps][::-1]
    y = np.vdot(w, x)
    if n < num_pilots:
        d = tx_symbols[n]
        e = d - y
        w += mu_train * np.conj(e) * x
    else:
        d = min_dist_detection(np.array([y]), const).ravel()[0]
        e = d - y
        w += mu_dd * np.conj(e) * x
    rx_lms[n] = y

rx_eq = rx_lms

rx_detected = min_dist_detection(rx_eq, const)

ser = calc_error_rate(tx_symbols, rx_detected)

plt.scatter(rx_eq.real, rx_eq.imag, color='blue', s=10, label='Received (equalized)')
plt.scatter(tx_symbols.real, tx_symbols.imag, color='orange', s=10, label='Transmitted')
plt.title(f'Received Symbols After Equalization\nSER={ser:.4f}')
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.savefig(str(LAB3_DIR / '5.png'))
plt.close()
