from ece132a import *
import matplotlib.pyplot as plt
from pathlib import Path

N = 1000

symbols_per_sec = 10**5
samples_per_symbol = 10

LAB3_DIR = Path(__file__).resolve().parent / "lab3"
LAB3_DIR.mkdir(exist_ok=True)
tx_path = LAB3_DIR / "tx_signal2.npy"
zc_path = LAB3_DIR / "zadoff_chu_sequence2.npy"
tx_payload_path = LAB3_DIR / "tx_payload_symbols2.npy"

# Always regenerate both sequences each run.
z = zadoff_chu()  # N=263,839
save_signal(str(zc_path.with_suffix("")), z)

const = get_const('64-QAM', 1)
tx_symbols = gen_rand_symbols(const, N, pad=False).ravel()
save_signal(str(tx_payload_path.with_suffix("")), tx_symbols)

symbols_padded = np.pad(tx_symbols, (int(N / 10), int(N / 10)), mode='constant')
# Prepend the Zadoff-Chu sequence for synchronization in later steps.
symbols_padded = np.concatenate([z.ravel(), symbols_padded.ravel()])
rc = get_rrc_pulse(0.5, 10, samples_per_symbol)
pulse_train = create_pulse_train(symbols_padded, samples_per_symbol)
tx_signal = np.convolve(pulse_train, rc, mode='same')
save_signal(str(tx_path.with_suffix("")), tx_signal)

tx_signal = load_signal(str(tx_path))

total_len = len(tx_signal)
t = np.linspace(0, total_len, total_len)

sample_idx = np.arange(0, total_len, samples_per_symbol)
plt.plot(sample_idx, tx_signal[sample_idx].real, 'o', color='orange', label='Real Samples')
plt.plot(sample_idx, tx_signal[sample_idx].imag, 'o', color='blue', label='Imag Samples')
plt.plot(t, np.real(tx_signal), color='orange', label='Real part')
plt.plot(t, np.imag(tx_signal), color='blue', label='Imaginary part')
plt.title('Transmit Signal (Zoomed In)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.savefig(str(LAB3_DIR / '1_extra.png'))
plt.xlim(460*samples_per_symbol, 540*samples_per_symbol)
plt.savefig(str(LAB3_DIR / '1.png'))
plt.close()
