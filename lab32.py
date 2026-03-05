from ece132a import *
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

N = 1000


symbols_per_sec = 10**5
samples_per_symbol = 10


LAB3_DIR = Path(__file__).resolve().parent / "lab3"
rx_path = LAB3_DIR / "rx_signal2.npy"
if not rx_path.exists():
    raise FileNotFoundError(
        f"Missing {rx_path}. Run `python3.10 sdr_lab3.py` first to capture RX data."
    )

rx_signal = load_signal(str(rx_path))
total_len = len(rx_signal)
t = np.linspace(0, total_len, total_len)

sample_idx = np.arange(0, total_len, samples_per_symbol)
plt.plot(sample_idx, rx_signal[sample_idx].real, 'ro', color='orange', label='Real Samples')
plt.plot(sample_idx, rx_signal[sample_idx].imag, 'ro', color='blue', label='Imag Samples')
plt.plot(t, np.real(rx_signal), color='orange', label='Real part')
plt.plot(t, np.imag(rx_signal), color='blue', label='Imaginary part')
plt.title('Recieved Signal (Zoomed In)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.savefig(str(LAB3_DIR / '2_extra.png'))
plt.xlim(460*samples_per_symbol, 540*samples_per_symbol)
plt.savefig(str(LAB3_DIR / '2.png'))
plt.close()
