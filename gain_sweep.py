from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np

from ece132a import (
    calc_error_rate,
    create_pulse_train,
    get_const,
    get_rrc_pulse,
    load_signal,
    min_dist_detection,
    save_signal,
)
from remoteRF.drivers.adalm_pluto import adi


LAB3_DIR = Path(__file__).resolve().parent / "lab3"
SWEEP_DIR = LAB3_DIR / "sweep_results"
LAB3_DIR.mkdir(exist_ok=True)
SWEEP_DIR.mkdir(exist_ok=True)

N = 1000
SPS = 10
BETA = 0.5
SPAN = 10
SAMPLE_RATE = int(1e6)
TX_LO_HZ = int(915e6)
RX_LO_HZ = int(915e6)
RX_BUFFER_SIZE = int(500e3)

TX_GAINS = [-35, -30, -25]
RX_GAINS = [20, 25, 30, 35]


def run_tx_rx_capture(tx_gain_db: int, rx_gain_db: int, token: str) -> np.ndarray:
    tx_signal = load_signal(str(LAB3_DIR / "tx_signal2.npy"))
    tx_signal_scaled = tx_signal / np.max(np.abs(tx_signal)) * (2**14)

    sdr = adi.Pluto(token=token)
    sdr.sample_rate = SAMPLE_RATE

    sdr.tx_destroy_buffer()
    sdr.tx_rf_bandwidth = SAMPLE_RATE
    sdr.tx_lo = TX_LO_HZ
    sdr.tx_hardwaregain_chan0 = tx_gain_db
    sdr.tx_cyclic_buffer = True

    sdr.rx_destroy_buffer()
    sdr.rx_lo = RX_LO_HZ
    sdr.rx_rf_bandwidth = SAMPLE_RATE
    sdr.rx_buffer_size = RX_BUFFER_SIZE
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = rx_gain_db

    sdr.tx(tx_signal_scaled)
    sdr.rx_destroy_buffer()
    _ = sdr.rx()
    rx_signal = sdr.rx()

    sdr.tx_destroy_buffer()
    sdr.rx_destroy_buffer()
    return rx_signal


def process_and_get_ser(rx_signal: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    z = load_signal(str(LAB3_DIR / "zadoff_chu_sequence2.npy"))
    tx_symbols = load_signal(str(LAB3_DIR / "tx_payload_symbols2.npy")).ravel()
    rrc = get_rrc_pulse(BETA, SPAN, SPS)

    zc = np.convolve(create_pulse_train(z, SPS), rrc, mode="same")
    corr = np.correlate(rx_signal, zc, mode="full")
    corr_mag = np.abs(corr)

    L = int((len(z) + (N + N * 2 / 10)) * SPS)
    lag_min = 0
    lag_max = len(rx_signal) - L
    if lag_max < lag_min:
        raise RuntimeError("RX buffer too short for one complete frame")

    idx_min = (len(zc) - 1) + lag_min
    idx_max = (len(zc) - 1) + lag_max
    offset = idx_min + np.argmax(corr_mag[idx_min:idx_max + 1])
    lag = offset - (len(zc) - 1)
    cor_rx_signal = rx_signal[lag:lag + L]

    mf_output = np.convolve(cor_rx_signal, rrc[::-1].conj(), mode="same")

    payload_start = (len(z) + int(N / 10)) * SPS
    const = get_const("64-QAM", 1)
    num_pilots = min(200, N // 2)
    pilot_tx = tx_symbols[:num_pilots]

    best_mse = np.inf
    best_omega = 0.0
    best_h = 1.0 + 0j
    best_rx_symbols = None

    for phase in range(SPS):
        cand = mf_output[payload_start + phase:payload_start + phase + N * SPS:SPS]
        if len(cand) != N:
            continue

        pilot_rx = cand[:num_pilots]
        ratio = pilot_rx / pilot_tx
        ang = np.unwrap(np.angle(ratio))
        n = np.arange(num_pilots)
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
            best_omega = omega
            best_h = h
            best_rx_symbols = cand

    if best_rx_symbols is None:
        raise RuntimeError("failed to recover symbols: no valid timing phase candidates")

    n_all = np.arange(N)
    rx_eq = best_rx_symbols * np.exp(-1j * best_omega * n_all) / best_h

    rx_hard = min_dist_detection(rx_eq, const).ravel()
    phase_err = np.angle(rx_eq * np.conj(rx_hard))
    theta = np.zeros_like(phase_err)
    alpha = 0.03
    for k in range(1, len(theta)):
        theta[k] = theta[k - 1] + alpha * phase_err[k]
    rx_eq = rx_eq * np.exp(-1j * theta)

    rx_detected = min_dist_detection(rx_eq, const)
    ser = calc_error_rate(tx_symbols, rx_detected)
    return ser, rx_eq, tx_symbols


def main() -> None:
    token = os.environ.get("HRgMuPDFjY8", "HRgMuPDFjY8")

    tx_file = LAB3_DIR / "tx_signal2.npy"
    zc_file = LAB3_DIR / "zadoff_chu_sequence2.npy"
    payload_file = LAB3_DIR / "tx_payload_symbols2.npy"
    for p in (tx_file, zc_file, payload_file):
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run `python3.10 lab31.py` first.")

    rows = []
    best = None

    for tx_gain in TX_GAINS:
        for rx_gain in RX_GAINS:
            rx_signal = run_tx_rx_capture(tx_gain, rx_gain, token)
            save_signal(str((LAB3_DIR / "rx_signal2").resolve()), rx_signal)
            ser, rx_eq, tx_symbols = process_and_get_ser(rx_signal)
            del rx_eq, tx_symbols
            rows.append({"tx_gain_db": tx_gain, "rx_gain_db": rx_gain, "ser": ser})
            print(f"SER={ser:.6f}")
            if best is None or ser < best["ser"]:
                best = {"tx_gain_db": tx_gain, "rx_gain_db": rx_gain, "ser": ser}
            time.sleep(0.25)

    csv_path = SWEEP_DIR / "gain_sweep_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tx_gain_db", "rx_gain_db", "ser"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results: {csv_path}")
    if best is not None:
        print(
            "Best setting: "
            f"TX={best['tx_gain_db']} dB, RX={best['rx_gain_db']} dB, SER={best['ser']:.6f}"
        )


if __name__ == "__main__":
    main()
