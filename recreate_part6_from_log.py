from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

LAB3_DIR = Path(__file__).resolve().parent / "lab3"
LAB3_DIR.mkdir(exist_ok=True)

TX_GAINS_DB = list(range(-50, -24))

# Part VI (with matched filter) data from your logged run.
SER_64QAM_MF = [
    0.961000, 0.813333, 0.798333, 0.639000, 0.576667, 0.610667, 0.493667,
    0.399000, 0.401333, 0.255667, 0.178000, 0.102333, 0.048667, 0.029667,
    0.028333, 0.012333, 0.003000, 0.001000, 0.003333, 0.021000, 0.004000,
    0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
]

SER_256QAM_MF = [
    0.935000, 0.902333, 0.904667, 0.889333, 0.841667, 0.808667, 0.780000,
    0.770333, 0.686000, 0.653667, 0.559333, 0.520000, 0.463667, 0.383000,
    0.284667, 0.184333, 0.213000, 0.166333, 0.133667, 0.146333, 0.073000,
    0.067333, 0.058333, 0.022333, 0.013000, 0.030000,
]


def load_part7_nomf_from_csv() -> tuple[list[float], list[float]]:
    part7_csv = LAB3_DIR / "7_no_matched_filter.csv"
    if not part7_csv.exists():
        raise FileNotFoundError(f"Missing {part7_csv}. Run lab37.py first.")

    ser64_nomf = []
    ser256_nomf = []
    gains = []

    with part7_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gains.append(int(float(row["tx_gain_db"])))
            ser64_nomf.append(float(row["ser_64qam_rc_no_mf"]))
            ser256_nomf.append(float(row["ser_256qam_rc_no_mf"]))

    if gains != TX_GAINS_DB:
        raise ValueError("Part VII gain axis does not match Part VI gain axis")

    return ser64_nomf, ser256_nomf


def save_overlay_plot(ser64_nomf: list[float], ser256_nomf: list[float]) -> None:
    plt.figure()
    plt.semilogy(TX_GAINS_DB, SER_64QAM_MF, "o-", label="64-QAM (RRC + MF)")
    plt.semilogy(TX_GAINS_DB, SER_256QAM_MF, "s-", label="256-QAM (RRC + MF)")
    plt.semilogy(TX_GAINS_DB, ser64_nomf, "o--", label="64-QAM (RC, no MF)")
    plt.semilogy(TX_GAINS_DB, ser256_nomf, "s--", label="256-QAM (RC, no MF)")
    plt.xlabel("Transmit Gain (dB)")
    plt.ylabel("SER")
    plt.title("Lab 3 Part VI + Part VII Overlay")
    plt.grid(True, which="both")
    plt.legend()
    plt.savefig(str(LAB3_DIR / "67_overlay_recreated.png"))
    plt.close()


def save_overlay_csv(ser64_nomf: list[float], ser256_nomf: list[float]) -> None:
    out = LAB3_DIR / "67_overlay_recreated.csv"
    with out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "tx_gain_db",
                "ser_64qam_mf",
                "ser_256qam_mf",
                "ser_64qam_rc_no_mf",
                "ser_256qam_rc_no_mf",
            ]
        )
        for row in zip(TX_GAINS_DB, SER_64QAM_MF, SER_256QAM_MF, ser64_nomf, ser256_nomf):
            writer.writerow(list(row))


def main() -> None:
    if len(TX_GAINS_DB) != len(SER_64QAM_MF) or len(TX_GAINS_DB) != len(SER_256QAM_MF):
        raise ValueError("Data length mismatch between gain axis and MF SER arrays")

    ser64_nomf, ser256_nomf = load_part7_nomf_from_csv()
    save_overlay_plot(ser64_nomf, ser256_nomf)
    save_overlay_csv(ser64_nomf, ser256_nomf)

    print("Saved:")
    print(LAB3_DIR / "67_overlay_recreated.png")
    print(LAB3_DIR / "67_overlay_recreated.csv")


if __name__ == "__main__":
    main()
