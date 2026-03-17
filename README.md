# ECE132A Lab 3 Notes

This repo contains the Lab 3 TX/RX processing chain for Pluto SDR experiments.

## What Was Changed (Brief)

- Added **RRC pulse support** in `ece132a.py` with `get_rrc_pulse(...)`.
- Updated TX/RX pipeline to use RRC pulse shaping and matched filtering:
  - `lab31.py` (TX shaping)
  - `lab33.py` (preamble correlation reference)
  - `lab34.py` (matched filter output)
- `lab31.py` now regenerates and saves:
  - `lab3/zadoff_chu_sequence2.npy`
  - `lab3/tx_signal2.npy`
  - `lab3/tx_payload_symbols2.npy`
- Improved frame sync in `lab33.py` using valid-lag peak selection for cyclic captures.
- `lab34.py` saves matched-filter output to `lab3/mf_output2.npy`.
- `lab35.py` updated receiver processing with:
  - pilot-based timing/CFO/channel estimation,
  - residual phase tracking,
  - **7-tap LMS equalizer** (pilot-trained then decision-directed).

## Current Parameters (In Code)

### Core DSP (`lab31.py`, `lab33.py`, `lab34.py`, `lab35.py`)
- `N = 1000`
- `symbols_per_sec = 1e5`
- `samples_per_symbol = 10`
- Pulse shaping: `RRC`
- `RRC beta = 0.5`
- `RRC span = 10`

### Receiver Tuning (`lab35.py`)
- `num_pilots = min(400, N // 2)`
- Phase tracker: `alpha = 0.01`
- LMS equalizer:
  - `eq_taps = 7`
  - `mu_train = 0.001`
  - `mu_dd = 0.0008`

### SDR Settings (`sdr_lab3.py`)
- `sample_rate = 1e6`
- `tx_carrier_freq_Hz = 915e6`
- `rx_carrier_freq_Hz = 915e6`
- `tx_gain_dB = -25`
- `rx_gain_dB = 45`
- `rx_agc_mode = 'manual'`
- `rx_buffer_size = 500e3`
- `tx_cyclic_buffer = True`

## Utility Scripts Added

- `run_lab3_pipeline.py`
  - Runs full chain in order:
    1. `lab31.py`
    2. `sdr_lab3.py`
    3. `lab33.py`
    4. `lab34.py`
    5. `lab35.py`

- `gain_sweep.py`
  - Sweeps TX/RX gains and prints **SER-only** lines.

- `ser_param_sweep.py`
  - Sweeps `alpha`, `mu_train`, `mu_dd` and prints SER + best combo.

## Current Typical Workflow

1. `python3.10 lab31.py`
2. `python3.10 sdr_lab3.py`
3. `python3.10 lab33.py`
4. `python3.10 lab34.py`
5. `python3.10 lab35.py`

Or run everything with:

```bash
python3.10 run_lab3_pipeline.py
```

## Notes

- SER can vary due to lab interference/channel changes.
- For fair comparisons, use multiple captures per setting and compare median SER.

## Two-SDR OFDM Message Demo (New)

Use `lab38_ofdm_two_sdr.py` to transmit a short UTF-8 message over OFDM using two Pluto SDRs.

### Features

- 64-point OFDM with CP length 16
- QPSK data on 48 subcarriers + 4 pilot tones
- Preamble-based frame sync
- One-tap channel estimate + pilot phase correction

### 1) Start TX SDR

Set your TX token and run:

```bash
export PLUTO_TX_TOKEN="<tx-token>"
python3.10 lab38_ofdm_two_sdr.py tx --message "HELLO FROM OFDM"
```

TX runs continuously (cyclic buffer) until `Ctrl+C`.

### 2) Capture + Decode on RX SDR

Set your RX token and run:

```bash
export PLUTO_RX_TOKEN="<rx-token>"
python3.10 lab38_ofdm_two_sdr.py rx
```

The script prints the decoded message and saves capture artifacts to `lab3/`:

- `lab3/ofdm_tx_frame.npy`
- `lab3/ofdm_rx_capture.npy`

### Optional Tuning

- TX gain: `--tx-gain-db` (example `-40` to `-25`)
- RX gain: `--rx-gain-db` (example `35` to `55`)
- Carrier frequency: `--carrier-hz`
- Sample rate: `--sample-rate`
