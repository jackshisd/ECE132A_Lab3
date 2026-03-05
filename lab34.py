from ece132a import *
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

N = 1000

symbols_per_sec = 10**5
samples_per_symbol = 10
LAB3_DIR = Path(__file__).resolve().parent / "lab3"
z_path = LAB3_DIR / "zadoff_chu_sequence2.npy"
cor_rx_path = LAB3_DIR / "cor_rx_signal2.npy"
mf_out_path = LAB3_DIR / "mf_output2.npy"
if not z_path.exists():
    raise FileNotFoundError(f"Missing {z_path}. Run `python3.10 lab31.py` first.")
if not cor_rx_path.exists():
    raise FileNotFoundError(
        f"Missing {cor_rx_path}. Run `python3.10 lab33.py` first."
    )

z = load_signal(str(z_path))
L = int((len(z) + (N+N*2/10))*samples_per_symbol)

rc = get_rrc_pulse(0.5, 10, samples_per_symbol) # matched RRC
rx_signal = load_signal(str(cor_rx_path)) # loads CORRELATED rx_signal of length L

# Matched filter: convolve rx_signal with time-reversed conjugate of transmit pulse (rc)
mf_output = np.convolve(rx_signal, rc[::-1].conj(), mode='same')
save_signal(str(mf_out_path.with_suffix("")), mf_output)

# Impulse response of the effective pulse shape (rc convolved with itself)
effective_pulse = np.convolve(rc, rc[::-1].conj(), mode='full')
effective_pulse /= np.max(np.abs(effective_pulse))  # normalize to peak 1

# Plotting the impulse response
plt.figure()
plt.plot(effective_pulse)
plt.title('Impulse Response of Effective Pulse Shape')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.savefig(str(LAB3_DIR / '41.png'))
plt.close()

# Real and imaginary parts of matched filter output
plt.figure()
plt.plot(np.real(mf_output), label='Real')
plt.plot(np.imag(mf_output), label='Imag')
plt.title('Matched Filter Output')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.savefig(str(LAB3_DIR / '42.png'))
plt.close()
