from ece132a import *
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

N = 1000

symbols_per_sec = 10**5
samples_per_symbol = 10

LAB3_DIR = Path(__file__).resolve().parent / "lab3"
z_path = LAB3_DIR / "zadoff_chu_sequence2.npy"
rx_path = LAB3_DIR / "rx_signal2.npy"
if not z_path.exists():
    raise FileNotFoundError(f"Missing {z_path}. Run `python3.10 lab31.py` first.")
if not rx_path.exists():
    raise FileNotFoundError(
        f"Missing {rx_path}. Run `python3.10 sdr_lab3.py` first to capture RX data."
    )

z = load_signal(str(z_path))
rc = get_rrc_pulse(0.5, 10, samples_per_symbol) # same as the one done on the transmitter
# Correlate against only the pulse-shaped preamble for robust frame start.
zc = np.convolve(create_pulse_train(z, samples_per_symbol), rc, mode='same')
rx_signal = load_signal(str(rx_path))

corr = np.correlate(rx_signal, zc, mode='full')
corr_mag = np.abs(corr)
plt.plot(corr_mag)
plt.xlabel('Sample Index')
plt.ylabel('Correlation Magnitude')
plt.savefig(str(LAB3_DIR / 'corr_mag.png'))
plt.grid(True)
plt.close()

L = int((len(z) + (N+N*2/10))*samples_per_symbol)
lag_min = 0
lag_max = len(rx_signal) - L
if lag_max < lag_min:
    raise RuntimeError("RX buffer too short for one complete frame")

# In cyclic TX captures there are many strong peaks. Restrict to peaks
# whose lag allows a full frame extraction, then choose the strongest.
idx_min = (len(zc) - 1) + lag_min
idx_max = (len(zc) - 1) + lag_max
offset = idx_min + np.argmax(corr_mag[idx_min:idx_max + 1])
lag = offset - (len(zc) - 1)
index_start = lag

cor_rx_signal = rx_signal[index_start:index_start + L] # new signal that has length L
save_signal(str((LAB3_DIR / 'cor_rx_signal2').resolve()), cor_rx_signal)

total_len = len(cor_rx_signal)
t = np.linspace(0, total_len, total_len)

plt.plot(t, np.abs(cor_rx_signal))
plt.xlabel('Sample Index')
plt.ylabel('Magnitude')
plt.title('Magnitude of Recieved Signal')
plt.grid(True)
plt.savefig(str(LAB3_DIR / '3.png'))
plt.close()
