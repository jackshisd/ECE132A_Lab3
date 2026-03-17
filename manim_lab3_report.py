from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from manim import *


LAB3_DIR = Path(__file__).resolve().parent / "lab3"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def fit_image_to_frame(img: ImageMobject, max_w: float = 12.5, max_h: float = 6.5) -> ImageMobject:
    if img.width > max_w:
        img.width = max_w
    if img.height > max_h:
        img.height = max_h
    return img


def downsample_xy(x: np.ndarray, y: np.ndarray, nmax: int = 1400) -> tuple[np.ndarray, np.ndarray]:
    n = len(x)
    if n <= nmax:
        return x, y
    idx = np.linspace(0, n - 1, nmax).astype(int)
    return x[idx], y[idx]


class Lab3ReportAnimation(Scene):
    """Slide-style sequence showing saved figures + overlay curve."""

    def show_part_image(self, part_title: str, filename: str, wait_t: float = 1.2) -> None:
        title = Text(part_title, font_size=36).to_edge(UP)
        path = LAB3_DIR / filename
        if not path.exists():
            missing = Text(f"Missing: {filename}", color=RED, font_size=28).move_to(ORIGIN)
            self.play(FadeIn(title), FadeIn(missing))
            self.wait(0.8)
            self.play(FadeOut(title), FadeOut(missing))
            return

        img = fit_image_to_frame(ImageMobject(str(path))).move_to(DOWN * 0.2)
        self.play(FadeIn(title), FadeIn(img, shift=UP * 0.2))
        self.wait(wait_t)
        self.play(FadeOut(title), FadeOut(img))

    def show_ser_overlay_from_csv(self) -> None:
        mf_csv = LAB3_DIR / "6_ser_vs_tx_gain_recreated.csv"
        nomf_csv = LAB3_DIR / "7_no_matched_filter.csv"

        mf_rows = read_csv_rows(mf_csv)
        nomf_rows = read_csv_rows(nomf_csv)
        if not mf_rows or not nomf_rows:
            self.show_part_image("Part 6/7 SER Curves", "67_overlay_recreated.png", wait_t=1.5)
            return

        x = [float(r["tx_gain_db"]) for r in mf_rows]
        y64_mf = [max(float(r["ser_64qam_mf"]), 1e-6) for r in mf_rows]
        y256_mf = [max(float(r["ser_256qam_mf"]), 1e-6) for r in mf_rows]

        nomf_map = {float(r["tx_gain_db"]): r for r in nomf_rows}
        y64_nomf = [max(float(nomf_map[g]["ser_64qam_rc_no_mf"]), 1e-6) for g in x]
        y256_nomf = [max(float(nomf_map[g]["ser_256qam_rc_no_mf"]), 1e-6) for g in x]

        title = Text("Part 6 + Part 7 SER Curves", font_size=34).to_edge(UP)
        axes = Axes(
            x_range=[-50, -25, 5],
            y_range=[0, 1.0, 0.2],
            x_length=10.5,
            y_length=5.5,
            axis_config={"font_size": 24},
            tips=False,
        ).move_to(DOWN * 0.35)

        x_label = Text("TX Gain (dB)", font_size=24).next_to(axes, DOWN)
        y_label = Text("SER", font_size=24).next_to(axes, LEFT)

        def mk_line(xv, yv, color):
            pts = [axes.c2p(xi, yi) for xi, yi in zip(xv, yv)]
            return VMobject(color=color).set_points_as_corners(pts)

        l64_mf = mk_line(x, y64_mf, BLUE)
        l256_mf = mk_line(x, y256_mf, GREEN)
        l64_nomf = mk_line(x, y64_nomf, YELLOW)
        l256_nomf = mk_line(x, y256_nomf, RED)

        self.play(FadeIn(title), Create(axes), FadeIn(x_label), FadeIn(y_label))
        self.play(Create(l64_mf), run_time=1.0)
        self.play(Create(l256_mf), run_time=1.0)
        self.play(Create(l64_nomf), run_time=1.0)
        self.play(Create(l256_nomf), run_time=1.0)
        self.wait(1.4)
        self.play(
            FadeOut(title), FadeOut(axes), FadeOut(x_label), FadeOut(y_label),
            FadeOut(l64_mf), FadeOut(l256_mf), FadeOut(l64_nomf), FadeOut(l256_nomf)
        )

    def construct(self):
        intro = Text("ECE 132A Lab 3: Parts 1-7", font_size=52)
        subtitle = Text("Results Walkthrough", font_size=30).next_to(intro, DOWN)
        self.play(Write(intro), FadeIn(subtitle, shift=UP * 0.2))
        self.wait(1.0)
        self.play(FadeOut(intro), FadeOut(subtitle))

        self.show_part_image("Part 1: Transmit Signal", "1.png")
        self.show_part_image("Part 2: Received Signal", "2.png")
        self.show_part_image("Part 3: Correlation / Frame Sync", "corr_mag.png")
        self.show_part_image("Part 3: Extracted Frame Magnitude", "3.png")
        self.show_part_image("Part 4: Effective Pulse Response", "41.png")
        self.show_part_image("Part 4: Matched Filter Output", "42.png")
        self.show_part_image("Part 5: Equalized Constellation", "5.png")
        self.show_part_image("Part 6: SER vs TX Gain", "6_ser_vs_tx_gain_recreated.png")
        self.show_part_image("Part 7: No Matched Filter", "7_no_matched_filter.png")
        self.show_ser_overlay_from_csv()


class LiveBuildLab3Animation(Scene):
    """Live-build animation: plots fill in real time from saved data."""

    def animate_line_build(self, x: np.ndarray, y: np.ndarray, title_text: str, x_label: str, y_label: str, run_time: float = 4.0) -> None:
        x, y = downsample_xy(x, y, nmax=1400)
        x_min, x_max = float(np.min(x)), float(np.max(x))
        y_min, y_max = float(np.min(y)), float(np.max(y))
        pad_y = 0.1 * (y_max - y_min + 1e-9)

        title = Text(title_text, font_size=34).to_edge(UP)
        axes = Axes(
            x_range=[x_min, x_max, max((x_max - x_min) / 5.0, 1e-6)],
            y_range=[y_min - pad_y, y_max + pad_y, max((y_max - y_min) / 5.0, 1e-6)],
            x_length=10.8,
            y_length=5.4,
            tips=False,
            axis_config={"font_size": 20},
        ).move_to(DOWN * 0.25)
        xl = Text(x_label, font_size=22).next_to(axes, DOWN)
        yl = Text(y_label, font_size=22).next_to(axes, LEFT)

        tracker = ValueTracker(2)

        def make_line():
            n = int(tracker.get_value())
            n = max(2, min(n, len(x)))
            pts = [axes.c2p(float(xi), float(yi)) for xi, yi in zip(x[:n], y[:n])]
            m = VMobject(color=BLUE)
            m.set_points_as_corners(pts)
            return m

        line = always_redraw(make_line)

        self.play(FadeIn(title), Create(axes), FadeIn(xl), FadeIn(yl))
        self.add(line)
        self.play(tracker.animate.set_value(len(x)), run_time=run_time, rate_func=linear)
        self.wait(0.6)
        self.play(FadeOut(title), FadeOut(axes), FadeOut(xl), FadeOut(yl), FadeOut(line))

    def animate_constellation_build(self, tx: np.ndarray, rx: np.ndarray) -> None:
        title = Text("Part 5: Constellation Build", font_size=34).to_edge(UP)
        lim = 1.6 * max(np.max(np.abs(tx.real)), np.max(np.abs(tx.imag)), 1.0)

        axes = Axes(
            x_range=[-lim, lim, lim / 4],
            y_range=[-lim, lim, lim / 4],
            x_length=7,
            y_length=7,
            tips=False,
            axis_config={"font_size": 20},
        ).move_to(DOWN * 0.1)

        nshow = min(600, len(rx))
        idx = np.linspace(0, len(rx) - 1, nshow).astype(int)
        rx_s = rx[idx]
        tx_s = tx[idx]

        rx_dots = VGroup(*[Dot(axes.c2p(float(r.real), float(r.imag)), radius=0.018, color=BLUE) for r in rx_s])
        tx_dots = VGroup(*[Dot(axes.c2p(float(t.real), float(t.imag)), radius=0.018, color=YELLOW) for t in tx_s])

        self.play(FadeIn(title), Create(axes))
        self.play(LaggedStart(*[FadeIn(d) for d in rx_dots], lag_ratio=0.002, run_time=3.2))
        self.play(LaggedStart(*[FadeIn(d) for d in tx_dots], lag_ratio=0.002, run_time=1.8))
        self.wait(0.8)
        self.play(FadeOut(title), FadeOut(axes), FadeOut(rx_dots), FadeOut(tx_dots))

    def animate_ser_build(self) -> None:
        mf_rows = read_csv_rows(LAB3_DIR / "6_ser_vs_tx_gain_recreated.csv")
        nomf_rows = read_csv_rows(LAB3_DIR / "7_no_matched_filter.csv")
        if not mf_rows or not nomf_rows:
            return

        x = [float(r["tx_gain_db"]) for r in mf_rows]
        y64_mf = [max(float(r["ser_64qam_mf"]), 1e-6) for r in mf_rows]
        y256_mf = [max(float(r["ser_256qam_mf"]), 1e-6) for r in mf_rows]
        nomf_map = {float(r["tx_gain_db"]): r for r in nomf_rows}
        y64_nomf = [max(float(nomf_map[g]["ser_64qam_rc_no_mf"]), 1e-6) for g in x]
        y256_nomf = [max(float(nomf_map[g]["ser_256qam_rc_no_mf"]), 1e-6) for g in x]

        title = Text("Part 6 + 7: SER Curves Build", font_size=34).to_edge(UP)
        axes = Axes(
            x_range=[-50, -25, 5],
            y_range=[0, 1.0, 0.2],
            x_length=10.5,
            y_length=5.5,
            tips=False,
            axis_config={"font_size": 20},
        ).move_to(DOWN * 0.3)

        self.play(FadeIn(title), Create(axes))

        def mk(points_x, points_y, color):
            pts = [axes.c2p(float(a), float(b)) for a, b in zip(points_x, points_y)]
            m = VMobject(color=color)
            m.set_points_as_corners(pts)
            return m

        self.play(Create(mk(x, y64_mf, BLUE)), run_time=1.0)
        self.play(Create(mk(x, y256_mf, GREEN)), run_time=1.0)
        self.play(Create(mk(x, y64_nomf, YELLOW)), run_time=1.0)
        self.play(Create(mk(x, y256_nomf, RED)), run_time=1.0)
        self.wait(1.0)
        self.play(FadeOut(title), FadeOut(axes))

    def construct(self):
        intro = Text("Live Data Build: Parts 1-7", font_size=48)
        self.play(Write(intro))
        self.wait(0.8)
        self.play(FadeOut(intro))

        tx = np.load(LAB3_DIR / "tx_signal2.npy")
        rx = np.load(LAB3_DIR / "rx_signal2.npy")
        corr_frame = np.load(LAB3_DIR / "cor_rx_signal2.npy")
        mf = np.load(LAB3_DIR / "mf_output2.npy")
        tx_payload = np.load(LAB3_DIR / "tx_payload_symbols2.npy")

        n1 = min(3500, len(tx))
        self.animate_line_build(np.arange(n1), np.real(tx[:n1]), "Part 1: TX Real", "Sample Index", "Amplitude", run_time=3.0)

        n2 = min(3500, len(rx))
        self.animate_line_build(np.arange(n2), np.real(rx[:n2]), "Part 2: RX Real", "Sample Index", "Amplitude", run_time=3.0)

        # Part 3 extracted frame magnitude
        mag = np.abs(corr_frame)
        n3 = min(3000, len(mag))
        self.animate_line_build(np.arange(n3), mag[:n3], "Part 3: Extracted Frame Magnitude", "Sample Index", "Magnitude", run_time=3.0)

        # Part 4 matched filter output (real)
        n4 = min(3500, len(mf))
        self.animate_line_build(np.arange(n4), np.real(mf[:n4]), "Part 4: Matched Filter Output (Real)", "Sample Index", "Amplitude", run_time=3.0)

        # Part 5 constellation build using payload and rough received symbols from MF stream
        start = int((len(np.load(LAB3_DIR / "zadoff_chu_sequence2.npy")) + int(1000 / 10)) * 10)
        rx_sym = mf[start:start + 1000 * 10:10]
        if len(rx_sym) >= 100:
            self.animate_constellation_build(tx_payload[:len(rx_sym)], rx_sym)

        self.animate_ser_build()

        out = Text("Animation Complete", font_size=40)
        self.play(Write(out))
        self.wait(0.8)
