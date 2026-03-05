from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ece132a import (
    calc_error_rate,
    create_pulse_train,
    gen_rand_symbols,
    get_const,
    get_rc_pulse,
    get_rrc_pulse,
    min_dist_detection,
    zadoff_chu,
)
from remoteRF.drivers.adalm_pluto import adi


LAB3_DIR = Path(__file__).resolve().parent / "lab3"
LAB3_DIR.mkdir(exist_ok=True)

N = 1000
SAMPLES_PER_SYMBOL = 10
SYMBOLS_PER_SEC = 10**5
NUM_PILOTS = min(400, N // 2)

RRC_BETA = 0.5
RRC_SPAN = 10

EQ_TAPS = 7
ALPHA = 0.01
MU_TRAIN = 0.001
MU_DD = 0.0008

SAMPLE_RATE = int(1e6)
TX_CARRIER_FREQ_HZ = int(915e6)
RX_CARRIER_FREQ_HZ = int(915e6)
RX_BUFFER_SIZE = int(500e3)
RX_GAIN_DB = 40
TX_GAINS_DB = list(range(-50, -24))
NUM_CAPTURES = 3

TOKEN = os.environ.get("PLUTO_TOKEN", "9RufX1Fm3-Q")


def setup_sdr(token: str) -> adi.Pluto:
    sdr = adi.Pluto(token=token)
    sdr.sample_rate = SAMPLE_RATE

    sdr.tx_destroy_buffer()
    sdr.tx_rf_bandwidth = SAMPLE_RATE
    sdr.tx_lo = TX_CARRIER_FREQ_HZ
    sdr.tx_cyclic_buffer = True

    sdr.rx_destroy_buffer()
    sdr.rx_lo = RX_CARRIER_FREQ_HZ
    sdr.rx_rf_bandwidth = SAMPLE_RATE
    sdr.rx_buffer_size = RX_BUFFER_SIZE
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = RX_GAIN_DB
    return sdr


def teardown_sdr(sdr: adi.Pluto) -> None:
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass


def _build_tx_waveform(modulation: str, use_rrc_tx: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    const = get_const(modulation, 1)
    tx_symbols = gen_rand_symbols(const, N, pad=False).ravel()
    z = zadoff_chu().ravel()

    pad = int(N / 10)
    payload = np.pad(tx_symbols, (pad, pad), mode="constant")
    frame_symbols = np.concatenate([z, payload])

    if use_rrc_tx:
        pulse = get_rrc_pulse(RRC_BETA, RRC_SPAN, SAMPLES_PER_SYMBOL)
    else:
        pulse = get_rc_pulse(RRC_BETA, RRC_SPAN, SAMPLES_PER_SYMBOL)

    pulse_train = create_pulse_train(frame_symbols, SAMPLES_PER_SYMBOL)
    tx_signal = np.convolve(pulse_train, pulse, mode="same")
    return tx_signal, tx_symbols, z, pulse


def _estimate_ser(
    rx_signal: np.ndarray,
    modulation: str,
    tx_symbols: np.ndarray,
    z: np.ndarray,
    pulse: np.ndarray,
    use_matched_filter: bool,
) -> float:
    const = get_const(modulation, 1)

    # Frame sync with known pulse-shaped ZC preamble.
    zc_ref = np.convolve(create_pulse_train(z, SAMPLES_PER_SYMBOL), pulse, mode="same")
    corr = np.correlate(rx_signal, zc_ref, mode="full")
    corr_mag = np.abs(corr)

    L = int((len(z) + (N + N * 2 / 10)) * SAMPLES_PER_SYMBOL)
    lag_min = 0
    lag_max = len(rx_signal) - L
    if lag_max < lag_min:
        raise RuntimeError("RX buffer too short for one complete frame")

    idx_min = (len(zc_ref) - 1) + lag_min
    idx_max = (len(zc_ref) - 1) + lag_max
    offset = idx_min + np.argmax(corr_mag[idx_min : idx_max + 1])
    lag = offset - (len(zc_ref) - 1)
    frame_rx = rx_signal[lag : lag + L]

    if use_matched_filter:
        rx_proc = np.convolve(frame_rx, pulse[::-1].conj(), mode="same")
    else:
        rx_proc = frame_rx

    payload_start = (len(z) + int(N / 10)) * SAMPLES_PER_SYMBOL
    pilot_tx = tx_symbols[:NUM_PILOTS]

    # Joint timing/CFO/channel estimation across sample phases.
    best_mse = np.inf
    best_omega = 0.0
    best_h = 1.0 + 0j
    best_symbols = None

    for phase in range(SAMPLES_PER_SYMBOL):
        cand = rx_proc[
            payload_start + phase : payload_start + phase + N * SAMPLES_PER_SYMBOL : SAMPLES_PER_SYMBOL
        ]
        if len(cand) != N:
            continue

        pilot_rx = cand[:NUM_PILOTS]
        ratio = pilot_rx / pilot_tx
        ang = np.unwrap(np.angle(ratio))
        n = np.arange(NUM_PILOTS)
        omega, _ = np.polyfit(n, ang, 1)

        n_all = np.arange(N)
        cand_derot = cand * np.exp(-1j * omega * n_all)
        pilot_derot = cand_derot[:NUM_PILOTS]

        den = np.vdot(pilot_tx, pilot_tx)
        if np.abs(den) < 1e-12:
            continue

        h = np.vdot(pilot_tx, pilot_derot) / den
        mse = np.mean(np.abs(pilot_derot - h * pilot_tx) ** 2)

        if mse < best_mse:
            best_mse = mse
            best_omega = omega
            best_h = h
            best_symbols = cand

    if best_symbols is None:
        raise RuntimeError("failed to recover symbols: no valid timing phase candidates")

    n_all = np.arange(N)
    rx_eq = best_symbols * np.exp(-1j * best_omega * n_all) / best_h

    # Residual phase tracking.
    hard = min_dist_detection(rx_eq, const).ravel()
    phase_err = np.angle(rx_eq * np.conj(hard))
    theta = np.zeros_like(phase_err)
    for k in range(1, len(theta)):
        theta[k] = theta[k - 1] + ALPHA * phase_err[k]
    rx_eq = rx_eq * np.exp(-1j * theta)

    # 7-tap LMS equalizer.
    half = EQ_TAPS // 2
    w = np.zeros(EQ_TAPS, dtype=np.complex128)
    w[half] = 1.0 + 0j
    rx_pad = np.pad(rx_eq, (half, half), mode="constant")
    rx_lms = np.zeros_like(rx_eq, dtype=np.complex128)

    for i in range(len(rx_eq)):
        x = rx_pad[i : i + EQ_TAPS][::-1]
        y = np.vdot(w, x)
        if i < NUM_PILOTS:
            d = tx_symbols[i]
            e = d - y
            w += MU_TRAIN * np.conj(e) * x
        else:
            d = min_dist_detection(np.array([y]), const).ravel()[0]
            e = d - y
            w += MU_DD * np.conj(e) * x
        rx_lms[i] = y

    detected = min_dist_detection(rx_lms, const)
    return calc_error_rate(tx_symbols, detected)


def run_ser_sweep(
    sdr: adi.Pluto,
    modulation: str,
    tx_gains_db: list[int],
    num_captures: int,
    use_rrc_tx: bool,
    use_matched_filter: bool,
) -> list[float]:
    tx_signal, tx_symbols, z, pulse = _build_tx_waveform(modulation, use_rrc_tx=use_rrc_tx)
    tx_scaled = tx_signal / np.max(np.abs(tx_signal)) * (2**14)

    ser_curve = []
    for tx_gain in tx_gains_db:
        sdr.tx_destroy_buffer()
        sdr.tx_hardwaregain_chan0 = tx_gain
        sdr.tx(tx_scaled)

        capture_sers = []
        for _ in range(num_captures):
            sdr.rx_destroy_buffer()
            _ = sdr.rx()  # flush one capture
            rx_signal = sdr.rx()
            ser = _estimate_ser(
                rx_signal,
                modulation=modulation,
                tx_symbols=tx_symbols,
                z=z,
                pulse=pulse,
                use_matched_filter=use_matched_filter,
            )
            capture_sers.append(ser)

        mean_ser = float(np.mean(capture_sers))
        ser_curve.append(mean_ser)
        print(
            f"SER={mean_ser:.6f} mod={modulation} tx_gain={tx_gain} "
            f"captures={num_captures} rrc_tx={use_rrc_tx} mf_rx={use_matched_filter}"
        )

    sdr.tx_destroy_buffer()
    return ser_curve


def _save_part6_plot(tx_gains_db: list[int], ser64: list[float], ser256: list[float]) -> None:
    plt.figure()
    plt.semilogy(tx_gains_db, ser64, "o-", label="64-QAM (RRC + MF)")
    plt.semilogy(tx_gains_db, ser256, "s-", label="256-QAM (RRC + MF)")
    plt.xlabel("Transmit Gain (dB)")
    plt.ylabel("SER")
    plt.title("Lab 3 Part VI: SER vs Transmit Gain")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(str(LAB3_DIR / "6_ser_vs_tx_gain.png"))
    plt.close()


def _save_part6_csv(tx_gains_db: list[int], ser64: list[float], ser256: list[float]) -> None:
    out = LAB3_DIR / "6_ser_vs_tx_gain.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tx_gain_db", "ser_64qam_mf", "ser_256qam_mf"])
        for g, s64, s256 in zip(tx_gains_db, ser64, ser256):
            writer.writerow([g, s64, s256])


def main() -> None:
    sdr = setup_sdr(TOKEN)
    try:
        ser64 = run_ser_sweep(
            sdr,
            modulation="64-QAM",
            tx_gains_db=TX_GAINS_DB,
            num_captures=NUM_CAPTURES,
            use_rrc_tx=True,
            use_matched_filter=True,
        )
        ser256 = run_ser_sweep(
            sdr,
            modulation="256-QAM",
            tx_gains_db=TX_GAINS_DB,
            num_captures=NUM_CAPTURES,
            use_rrc_tx=True,
            use_matched_filter=True,
        )
    finally:
        teardown_sdr(sdr)

    _save_part6_plot(TX_GAINS_DB, ser64, ser256)
    _save_part6_csv(TX_GAINS_DB, ser64, ser256)


if __name__ == "__main__":
    main()
