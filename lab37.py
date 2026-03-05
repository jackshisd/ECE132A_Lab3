from __future__ import annotations

import csv

import matplotlib.pyplot as plt

from lab36 import (
    LAB3_DIR,
    NUM_CAPTURES,
    TOKEN,
    TX_GAINS_DB,
    run_ser_sweep,
    setup_sdr,
    teardown_sdr,
)


def _save_part7_nomf_plot(tx_gains_db, ser64_nomf, ser256_nomf):
    plt.figure()
    plt.semilogy(tx_gains_db, ser64_nomf, "o--", label="64-QAM (RC, no MF)")
    plt.semilogy(tx_gains_db, ser256_nomf, "s--", label="256-QAM (RC, no MF)")
    plt.xlabel("Transmit Gain (dB)")
    plt.ylabel("SER")
    plt.title("Lab 3 Part VII: No Matched Filter (RC TX)")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(str(LAB3_DIR / "7_no_matched_filter.png"))
    plt.close()


def _save_part7_nomf_csv(tx_gains_db, ser64_nomf, ser256_nomf):
    out = LAB3_DIR / "7_no_matched_filter.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tx_gain_db", "ser_64qam_rc_no_mf", "ser_256qam_rc_no_mf"])
        for row in zip(tx_gains_db, ser64_nomf, ser256_nomf):
            writer.writerow(list(row))


def main() -> None:
    sdr = setup_sdr(TOKEN)
    try:
        # Part VII variant only: RC TX + no matched filter RX
        ser64_nomf = run_ser_sweep(
            sdr,
            modulation="64-QAM",
            tx_gains_db=TX_GAINS_DB,
            num_captures=NUM_CAPTURES,
            use_rrc_tx=False,
            use_matched_filter=False,
        )
        ser256_nomf = run_ser_sweep(
            sdr,
            modulation="256-QAM",
            tx_gains_db=TX_GAINS_DB,
            num_captures=NUM_CAPTURES,
            use_rrc_tx=False,
            use_matched_filter=False,
        )
    finally:
        teardown_sdr(sdr)

    _save_part7_nomf_plot(TX_GAINS_DB, ser64_nomf, ser256_nomf)
    _save_part7_nomf_csv(TX_GAINS_DB, ser64_nomf, ser256_nomf)


if __name__ == "__main__":
    main()
