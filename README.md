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
- Tuned receiver params in `lab35.py`:
  - `alpha = 0.01`
  - `mu_train = 0.001`
  - `mu_dd = 0.0008`
  - `num_pilots = min(400, N // 2)`

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
