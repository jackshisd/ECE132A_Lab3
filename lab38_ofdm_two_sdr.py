from __future__ import annotations

import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from remoteRF.drivers.adalm_pluto import adi


LAB3_DIR = Path(__file__).resolve().parent / "lab3"
LAB3_DIR.mkdir(exist_ok=True)

# Set these manually for your two Pluto devices.
TX_TOKEN_MANUAL = "Ibx87-7BfLY"
RX_TOKEN_MANUAL = "Q_6mAsLyZ1w"

PACKET_MAGIC = b"PKT1"


@dataclass
class LinkConfig:
    sample_rate: int = int(1e6)
    carrier_hz: int = int(915e6)
    tx_gain_db: float = -35.0
    rx_gain_db: float = 45.0
    rx_buffer_size: int = int(250_000)
    sps: int = 10


def _bytes_to_bits(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8)).astype(np.uint8)


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    n = (bits.size // 8) * 8
    if n <= 0:
        return b""
    return np.packbits(bits[:n].astype(np.uint8)).tobytes()


def _dbpsk_mod(bits: np.ndarray) -> np.ndarray:
    bits = bits.astype(np.uint8)
    out = np.empty(bits.size, dtype=np.complex128)
    phase = 0.0
    for i, b in enumerate(bits):
        if b == 1:
            phase += np.pi
        out[i] = np.exp(1j * phase)
    return out


def _dbpsk_demod(symbols: np.ndarray) -> np.ndarray:
    if symbols.size < 2:
        return np.zeros(0, dtype=np.uint8)
    d = symbols[1:] * np.conj(symbols[:-1])
    return (np.real(d) < 0).astype(np.uint8)


def _upsample_rect(symbols: np.ndarray, sps: int) -> np.ndarray:
    return np.repeat(symbols, sps)


def _pack_message_bits(message: str) -> np.ndarray:
    payload = message.encode("utf-8")
    if len(payload) > 1023:
        raise ValueError("message too long; keep <= 1023 bytes")

    header = PACKET_MAGIC + len(payload).to_bytes(2, "big")
    crc = (zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "big")
    return _bytes_to_bits(header + payload + crc)


def _build_tx_waveform(message: str, cfg: LinkConfig) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(132)
    preamble_bits = rng.integers(0, 2, size=128, dtype=np.uint8)
    packet_bits = _pack_message_bits(message)

    # Repeat full frame to make capture timing simple.
    frame_bits = np.concatenate([preamble_bits, packet_bits])
    stream_bits = np.tile(frame_bits, 40)

    tx_syms = _dbpsk_mod(stream_bits)
    tx = _upsample_rect(tx_syms, cfg.sps)
    tx = tx / (np.max(np.abs(tx)) + 1e-12)

    np.save(LAB3_DIR / "dbpsk_preamble_bits.npy", preamble_bits)
    np.save(LAB3_DIR / "dbpsk_frame_bits.npy", frame_bits)
    np.save(LAB3_DIR / "dbpsk_tx_waveform.npy", tx.astype(np.complex64))
    return tx.astype(np.complex64), len(packet_bits)


def _try_decode_bits(hard_bits: np.ndarray, preamble_bits: np.ndarray, max_payload_bytes: int = 1023) -> str | None:
    if hard_bits.size < preamble_bits.size + 80:
        return None

    a = 2 * hard_bits.astype(np.int16) - 1
    b = 2 * preamble_bits.astype(np.int16) - 1
    corr = np.correlate(a, b, mode="valid")
    if corr.size == 0:
        return None

    # Check several strongest alignment candidates.
    order = np.argsort(corr)[::-1]
    top = order[:20]

    for start in top:
        if corr[start] < 0.65 * preamble_bits.size:
            continue

        pos = start + preamble_bits.size

        # Header = magic(4) + length(2)
        hdr_bits_len = (4 + 2) * 8
        if pos + hdr_bits_len > hard_bits.size:
            continue

        hdr = _bits_to_bytes(hard_bits[pos : pos + hdr_bits_len])
        if len(hdr) != 6:
            continue
        if hdr[:4] != PACKET_MAGIC:
            continue

        msg_len = int.from_bytes(hdr[4:6], "big")
        if msg_len < 0 or msg_len > max_payload_bytes:
            continue

        total_bits = hdr_bits_len + (msg_len * 8) + 32
        if pos + total_bits > hard_bits.size:
            continue

        body = _bits_to_bytes(hard_bits[pos + hdr_bits_len : pos + total_bits])
        if len(body) < msg_len + 4:
            continue

        payload = body[:msg_len]
        rx_crc = body[msg_len : msg_len + 4]
        calc_crc = (zlib.crc32(payload) & 0xFFFFFFFF).to_bytes(4, "big")
        if rx_crc != calc_crc:
            continue

        return payload.decode("utf-8", errors="replace")

    return None


def _decode_capture(rx: np.ndarray, cfg: LinkConfig) -> str | None:
    preamble_bits = np.load(LAB3_DIR / "dbpsk_preamble_bits.npy")

    # Normalize and mild smoothing to improve symbol decisions.
    rx = rx.astype(np.complex128)
    rx = rx / (np.max(np.abs(rx)) + 1e-12)
    smooth = np.convolve(rx, np.ones(cfg.sps) / cfg.sps, mode="same")

    for phase in range(cfg.sps):
        syms = smooth[phase:: cfg.sps]
        bits = _dbpsk_demod(syms)
        msg = _try_decode_bits(bits, preamble_bits)
        if msg is not None:
            return msg

    return None


def _setup_tx_sdr(token: str, cfg: LinkConfig) -> adi.Pluto:
    sdr = adi.Pluto(token=token)
    sdr.sample_rate = int(cfg.sample_rate)

    sdr.tx_destroy_buffer()
    sdr.tx_rf_bandwidth = int(cfg.sample_rate)
    sdr.tx_lo = int(cfg.carrier_hz)
    sdr.tx_hardwaregain_chan0 = float(cfg.tx_gain_db)
    sdr.tx_cyclic_buffer = True
    return sdr


def _setup_rx_sdr(token: str, cfg: LinkConfig) -> adi.Pluto:
    sdr = adi.Pluto(token=token)
    sdr.sample_rate = int(cfg.sample_rate)

    sdr.rx_destroy_buffer()
    sdr.rx_lo = int(cfg.carrier_hz)
    sdr.rx_rf_bandwidth = int(cfg.sample_rate)
    sdr.rx_buffer_size = int(cfg.rx_buffer_size)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.rx_hardwaregain_chan0 = float(cfg.rx_gain_db)
    return sdr


def _teardown_sdr(sdr: adi.Pluto) -> None:
    try:
        sdr.tx_destroy_buffer()
    except Exception:
        pass
    try:
        sdr.rx_destroy_buffer()
    except Exception:
        pass


def run_once() -> None:
    cfg = LinkConfig()

    if not TX_TOKEN_MANUAL or not RX_TOKEN_MANUAL:
        raise ValueError("set TX_TOKEN_MANUAL and RX_TOKEN_MANUAL in this file")

    message = input("Enter message to transmit: ").strip()
    if not message:
        raise ValueError("message cannot be empty")

    tx_waveform, packet_bits = _build_tx_waveform(message, cfg)
    tx_scaled = tx_waveform / (np.max(np.abs(tx_waveform)) + 1e-12) * (2**14)

    print("Connecting to TX and RX SDRs...", flush=True)
    tx_sdr = _setup_tx_sdr(TX_TOKEN_MANUAL, cfg)
    rx_sdr = _setup_rx_sdr(RX_TOKEN_MANUAL, cfg)

    rx_best = None
    decoded = None

    trial_tx_gains = [-40.0, -35.0, -30.0, -25.0]
    trial_rx_gains = [35.0, 40.0, 45.0, 50.0, 55.0]

    try:
        total = len(trial_tx_gains) * len(trial_rx_gains)
        k = 0

        for tx_gain in trial_tx_gains:
            tx_sdr.tx_destroy_buffer()
            tx_sdr.tx_hardwaregain_chan0 = float(tx_gain)
            tx_sdr.tx(tx_scaled)
            time.sleep(0.2)

            for rx_gain in trial_rx_gains:
                k += 1
                print(f"Attempt {k}/{total}: tx={tx_gain} dB, rx={rx_gain} dB", flush=True)
                rx_sdr.rx_hardwaregain_chan0 = float(rx_gain)

                rx_sdr.rx_destroy_buffer()
                _ = rx_sdr.rx()
                rx = rx_sdr.rx()
                np.save(LAB3_DIR / "dbpsk_rx_capture.npy", rx)

                msg = _decode_capture(rx, cfg)
                if msg is not None:
                    rx_best = rx
                    decoded = msg
                    print(f"  decoded: {msg}", flush=True)
                    if msg == message:
                        print(f"Found working gains: tx={tx_gain} dB, rx={rx_gain} dB", flush=True)
                        break
                else:
                    print("  decode failed", flush=True)

            if decoded == message:
                break

    finally:
        _teardown_sdr(tx_sdr)
        _teardown_sdr(rx_sdr)

    if decoded != message:
        raise RuntimeError("no exact message match found; adjust antenna placement and rerun")

    print(f"Transmitted packet bits: {packet_bits}")
    print(f"Transmitted: {message}")
    print(f"Decoded:     {decoded}")
    print("Status: SUCCESS (exact match)")


if __name__ == "__main__":
    run_once()
