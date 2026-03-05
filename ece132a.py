import numpy as np


def zadoff_chu(N=127, q=1):
    n = np.arange(N)
    return np.exp(-1j * np.pi * q * n * (n) / N) # N MUST BE ODD

def save_gen_rand_qam_symbols(path, symbols, constellation):
    np.savez(path, symbols=symbols, constellation=constellation)

def load_gen_rand_qam_symbols(path):
    d = np.load(path, allow_pickle=True)
    if hasattr(d, 'files'):
        try:
            return d['symbols'], d['constellation']
        except KeyError:
            raise ValueError("file does not contain 'symbols' and 'constellation' entries")
    else:
        raise ValueError("expected an .npz archive created by save_gen_rand_qam_symbols")

def save_signal(path, rx_signal):
    np.save(path, rx_signal)

def load_signal(path):
    return np.load(path, allow_pickle=True)

def gen_rand_qam_symbols(N, M=4):

    if M <= 0 or int(np.sqrt(M))**2 != M:
        raise ValueError("M must be a positive perfect square (e.g. 4,16,64).")

    m = int(np.sqrt(M))
    levels = np.arange(-m + 1, m, 2)
    a, b = np.meshgrid(levels, levels)
    const_points = (a.flatten() + 1j * b.flatten()).astype(np.complex128)
    avg_energy = np.mean(np.abs(const_points)**2)
    const_points = const_points / np.sqrt(avg_energy)

    idx = np.random.randint(0, M, size=N)
    symbols = const_points[idx]
    return symbols, const_points


def create_pulse_train(symbols, sps):
    symbols = np.asarray(symbols)
    if symbols.ndim != 1:
        symbols = symbols.ravel()
    if not (isinstance(sps, (int, np.integer)) and sps >= 1):
        raise ValueError("sps must be an integer >= 1")

    out = np.zeros(symbols.size * sps, dtype=symbols.dtype)
    out[::sps] = symbols
    return out


def get_rc_pulse(beta, span, sps):
    if not (0 <= beta <= 1):
        raise ValueError("beta must be between 0 and 1 inclusive")
    if not (isinstance(span, (int, np.integer)) and span >= 0):
        raise ValueError("span must be a non-negative integer")
    if not (isinstance(sps, (int, np.integer)) and sps >= 1):
        raise ValueError("sps must be an integer >= 1")

    N = int(span * sps) + 1
    t = np.linspace(-span / 2.0, span / 2.0, N)

    if beta == 0:
        p = np.sinc(t)
    else:
        x = t
        pi = np.pi
        num = np.sinc(x)
        cos_term = np.cos(pi * beta * x)
        denom = 1.0 - (2.0 * beta * x) ** 2

        p = num * cos_term / denom

        singular = np.isclose(np.abs(2.0 * beta * x), 1.0)
        if np.any(singular):
            lim_val = (np.pi / 4.0) * np.sinc(1.0 / (2.0 * beta))
            p[singular] = lim_val

    energy = np.sum(np.abs(p) ** 2) / float(sps)
    if energy <= 0:
        raise RuntimeError("computed pulse has non-positive energy")
    p = p / np.sqrt(energy) # Normalized for unit energy
    return p


def get_rrc_pulse(beta, span, sps):
    if not (0 <= beta <= 1):
        raise ValueError("beta must be between 0 and 1 inclusive")
    if not (isinstance(span, (int, np.integer)) and span >= 0):
        raise ValueError("span must be a non-negative integer")
    if not (isinstance(sps, (int, np.integer)) and sps >= 1):
        raise ValueError("sps must be an integer >= 1")

    N = int(span * sps) + 1
    t = np.linspace(-span / 2.0, span / 2.0, N)

    if beta == 0:
        p = np.sinc(t)
    else:
        pi = np.pi
        num = np.sin(pi * t * (1.0 - beta)) + 4.0 * beta * t * np.cos(pi * t * (1.0 + beta))
        den = pi * t * (1.0 - (4.0 * beta * t) ** 2)
        p = np.divide(num, den, out=np.zeros_like(t, dtype=np.float64), where=~np.isclose(den, 0.0))

        zero_idx = np.isclose(t, 0.0)
        if np.any(zero_idx):
            p[zero_idx] = 1.0 - beta + (4.0 * beta / pi)

        t0 = 1.0 / (4.0 * beta)
        edge_idx = np.isclose(np.abs(t), t0)
        if np.any(edge_idx):
            edge_val = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / pi) * np.sin(pi / (4.0 * beta))
                + (1.0 - 2.0 / pi) * np.cos(pi / (4.0 * beta))
            )
            p[edge_idx] = edge_val

    energy = np.sum(np.abs(p) ** 2) / float(sps)
    if energy <= 0:
        raise RuntimeError("computed pulse has non-positive energy")
    p = p / np.sqrt(energy)
    return p


def get_const(digmod, Es):
    # returns a constellation of symbols S ∈ C^(M×1) based on the input string digmod. For instance, digmod may be ‘BPSK’, ‘16-QAM’, etc. You should support OOK, BPSK, QPSK, M-PSK, M-PAM, and M-QAM. The returned constellation should have an average energy per symbol of Es.
    dm = str(digmod).strip().upper()
    if dm == 'OOK':
        s = np.array([0.0, 1.0], dtype=np.complex128)
        s = s/np.sqrt(np.mean(np.abs(s)**2)) * np.sqrt(Es)
        return s.reshape(-1, 1)
    if dm == 'BPSK':
        s = np.array([-1.0, 1.0], dtype=np.complex128)
        s = s/np.sqrt(np.mean(np.abs(s)**2)) * np.sqrt(Es)
        return s.reshape(-1, 1)
    if dm == 'QPSK':
        s = np.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=np.complex128)
        s = s/np.sqrt(np.mean(np.abs(s)**2)) * np.sqrt(Es)
        return s.reshape(-1, 1)
    if 'PSK' in dm:
        import re
        m = re.search(r"(\d+)", dm)
        M = int(m.group(1)) if m else 8
        k = np.arange(M)
        s = np.exp(1j * 2.0 * np.pi * k / M).astype(np.complex128)
        s = s/np.sqrt(np.mean(np.abs(s)**2)) * np.sqrt(Es)
        return s.reshape(-1, 1)
    if 'PAM' in dm:
        import re
        m = re.search(r"(\d+)", dm)
        M = int(m.group(1)) if m else 4
        levels = np.arange(-M + 1, M, 2)
        s = levels.astype(np.complex128)
        s = s/np.sqrt(np.mean(np.abs(s)**2)) * np.sqrt(Es)
        return s.reshape(-1, 1)
    if 'QAM' in dm:
        import re
        m = re.search(r"(\d+)", dm)
        M = int(m.group(1)) if m else 16
        mroot = int(np.sqrt(M))
        levels = np.arange(-mroot + 1, mroot, 2)
        a, b = np.meshgrid(levels, levels)
        s = (a.flatten() + 1j * b.flatten()).astype(np.complex128)
        s = s/np.sqrt(np.mean(np.abs(s)**2)) * np.sqrt(Es)
        return s.reshape(-1, 1)
    return np.zeros((0, 1), dtype=np.complex128)

def get_const_metrics(S):
    # returns the average energy per symbol Es, the minimum distance dmin, and the modulation order M of a given constellation of symbols S ∈ C^(M×1)
    # in form [Es, dmin, M]
    s = np.asarray(S).ravel()
    M = s.size
    Es = float(np.mean(np.abs(s) ** 2))
    if M <= 1:
        dmin = 0.0
    else:
        dists = np.abs(s.reshape(-1, 1) - s.reshape(1, -1))
        np.fill_diagonal(dists, np.inf)
        dmin = float(np.min(dists))
    return [Es, dmin, M]
                      
def gen_rand_symbols(S, N, pad=False):
    # returns a vector s ∈ C^(N×1) of N elements drawn at random (uniformly) from the constellation of M complex symbols S ∈ C^(M×1)
    s = np.asarray(S).ravel()
    M = s.size
    idx = np.random.randint(0, M, size=int(N))
    out = s[idx].astype(np.complex128)
    if pad:
        out = np.pad(out, (int(N/10), int(N/10)), mode = 'constant')
    return out.reshape(-1, 1)

def min_dist_detection(y, S):
    # takes as input a vector y of N complex values y ∈ C^(N×1) and maps each to the nearest element in the vector of M complex symbols S ∈ C^(M×1). Here, “nearest” refers to the minimum Euclidean distance
    yv = np.asarray(y).ravel()
    s = np.asarray(S).ravel()
    d = np.abs(yv.reshape(-1, 1) - s.reshape(1, -1))
    idx = np.argmin(d, axis=1)
    out = s[idx].astype(np.complex128)
    return out.reshape(-1, 1)

def calc_error_rate(s1, s2):
    # returns the fraction of entries that differ between two sequences of N values, s1 ∈ C^(N×1) and s2 ∈ C^(N×1)
    a = np.asarray(s1).ravel()
    b = np.asarray(s2).ravel()
    n = min(a.size, b.size)
    if n == 0:
        return 0.0
    diff = np.sum(a[:n] != b[:n])
    return float(diff) / float(n)
