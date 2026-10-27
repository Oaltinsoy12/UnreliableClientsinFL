#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute uplink SINR / SNR for an over-the-air FL round from saved local weight updates.

Expects, per run folder, one or both of these file pairs:
  <run>/first_w_locals.pt         + <run>/latest_w_locals.pt         (2-malicious variant)
  <run>/first_w_locals_1user.pt   + <run>/latest_w_locals_1user.pt   (1-malicious variant)

first_w_locals*.pt  : list of N OrderedDict state_dicts, ALL users (incl. malicious) -> pre-defense
latest_w_locals*.pt : list of M OrderedDict state_dicts, reliable users ONLY (M < N) -> post-defense

Two different metrics, because "latest" has already had the malicious users excluded:
  SINR (from first_w_locals*)  = P_signal(reliable) / (P_interference(malicious) + P_noise)
  SNR  (from latest_w_locals*) = P_signal(reliable) / P_noise                      (no interference term
                                                                                     -- malicious already excluded)

Malicious users are identified as the (N - M) entries in first_w_locals* whose
nearest neighbour (by L2 distance) among latest_w_locals* is farthest away.

Usage:
  python sinr.py 001_8 003_8 005_8
  python sinr.py 001_8 --noise-std 0.01
  python sinr.py 001_8 --malicious-idx 4,5
  python sinr.py 001_8 --gains gains.npy      # optional per-user channel gain |h_k|, unit gain if omitted
  python sinr.py 001_8 --verbose              # show P_signal / P_interference / P_noise breakdown
  python sinr.py 001_8 003_8 005_8 001_16 003_16 005_16 --plot sinr_comparison.png
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

# Categorical palette (validated, fixed order — see dataviz skill palette.md).
# Assigned by (metric, variant) so the same series always gets the same color everywhere.
SERIES_COLORS = {
    ("SINR", "2-malicious"): "#2a78d6",  # slot 1 blue
    ("SNR",  "2-malicious"): "#eb6834",  # slot 2 orange
    ("SINR", "1-malicious"): "#1baf7a",  # slot 3 aqua
    ("SNR",  "1-malicious"): "#eda100",  # slot 4 yellow
}
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"

VARIANTS = [
    ("2-malicious", ""),
    ("1-malicious", "_1user"),
]


def flatten(state_dict):
    return torch.cat([v.flatten().float() for v in state_dict.values()])


def load_locals(run_dir, suffix=""):
    first_path = os.path.join(run_dir, f"first_w_locals{suffix}.pt")
    latest_path = os.path.join(run_dir, f"latest_w_locals{suffix}.pt")
    if not (os.path.exists(first_path) and os.path.exists(latest_path)):
        return None
    first = torch.load(first_path, map_location="cpu", weights_only=False)
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    first_flat = [flatten(sd) for sd in first]
    latest_flat = [flatten(sd) for sd in latest]
    return first_flat, latest_flat


def detect_malicious(first_flat, latest_flat):
    n_malicious = len(first_flat) - len(latest_flat)
    if n_malicious <= 0:
        raise ValueError(
            f"first_w_locals has {len(first_flat)} users but latest_w_locals has "
            f"{len(latest_flat)} — expected latest to be a strict subset (reliable-only)."
        )
    min_dists = [min(torch.norm(f - l).item() for l in latest_flat) for f in first_flat]
    # the n_malicious entries farthest from any reliable update are malicious
    ranked = sorted(range(len(first_flat)), key=lambda i: min_dists[i], reverse=True)
    malicious_idx = sorted(ranked[:n_malicious])
    return malicious_idx


def load_gains(gains_arg, n_users):
    if gains_arg is None:
        return np.ones(n_users)
    if gains_arg.endswith(".npy"):
        gains = np.load(gains_arg)
    else:
        gains = np.array([float(x) for x in gains_arg.split(",")])
    if len(gains) != n_users:
        raise ValueError(f"Expected {n_users} channel gains, got {len(gains)}")
    return gains


def compute_sinr(first_flat, malicious_idx, noise_std, gains):
    """Pre-defense: signal (reliable, coherent) vs interference (malicious, incoherent) + noise.

    Noise is applied independently to each user's weights *before* saving/combining, so when M
    reliable users are summed coherently, their independent noise terms add incoherently: the
    aggregate noise power is sum(gain_i^2 * noise_std^2) over the reliable users, not noise_std^2
    on its own. That noise energy is also baked into the raw observed signal power, so it must be
    subtracted back out before treating the raw power as "signal".
    """
    d = first_flat[0].numel()
    reliable_idx = [i for i in range(len(first_flat)) if i not in malicious_idx]

    signal_sum = torch.zeros_like(first_flat[0])
    for i in reliable_idx:
        signal_sum += gains[i] * first_flat[i]
    p_signal_raw = (signal_sum.norm() ** 2 / d).item()

    p_noise = sum(gains[i] ** 2 for i in reliable_idx) * (noise_std ** 2)
    p_signal = max(p_signal_raw - p_noise, 0.0)

    # independent malicious users -> powers add incoherently (kept raw: interference power is
    # whatever total power arrives from that source, noise-on-attacker included)
    p_interference = sum(((gains[i] * first_flat[i]).norm() ** 2 / d).item() for i in malicious_idx)

    sinr = p_signal / (p_interference + p_noise)
    return {
        "P_signal_raw": p_signal_raw,
        "P_signal": p_signal,
        "P_interference": p_interference,
        "P_noise": p_noise,
        "SINR": sinr,
        "SINR_dB": 10 * np.log10(sinr),
    }


def compute_snr(latest_flat, noise_std):
    """Post-defense: malicious users already excluded from latest_w_locals -> signal vs noise only.

    Same incoherent-noise-summing correction as compute_sinr, over the M users in latest_flat.
    """
    d = latest_flat[0].numel()
    M = len(latest_flat)
    signal_sum = torch.zeros_like(latest_flat[0])
    for l in latest_flat:
        signal_sum += l
    p_signal_raw = (signal_sum.norm() ** 2 / d).item()

    p_noise = M * (noise_std ** 2)
    p_signal = max(p_signal_raw - p_noise, 0.0)

    snr = p_signal / p_noise
    return {
        "P_signal_raw": p_signal_raw,
        "P_signal": p_signal,
        "P_noise": p_noise,
        "SNR": snr,
        "SNR_dB": 10 * np.log10(snr),
    }


def compute_sinr_closed_form(first_flat, malicious_idx, noise_std, gains):
    """Scalar closed-form approximation (see README "Closed-form scalar approximation"):

    SINR = [((n-N_malicious)*w)^2 - P_noise] / (N_malicious*w_malicious^2 + P_noise)

    where w / w_malicious are per-element RMS amplitudes of a "typical" reliable / malicious
    user (sqrt of mean per-user power). Assumes all reliable vectors are perfectly coherent
    (‖sum w_k‖ = sum ‖w_k‖), so this is an upper-bound idealization of compute_sinr().
    """
    d = first_flat[0].numel()
    reliable_idx = [i for i in range(len(first_flat)) if i not in malicious_idx]
    M, N_mal = len(reliable_idx), len(malicious_idx)

    rel_powers = [(gains[i] ** 2 * first_flat[i].norm() ** 2 / d).item() for i in reliable_idx]
    mal_powers = [(gains[i] ** 2 * first_flat[i].norm() ** 2 / d).item() for i in malicious_idx]
    w = np.sqrt(np.mean(rel_powers))
    w_mal = np.sqrt(np.mean(mal_powers)) if mal_powers else 0.0

    p_noise = M * (noise_std ** 2)
    p_signal_raw = (M * w) ** 2
    p_signal = max(p_signal_raw - p_noise, 0.0)
    p_interference = N_mal * (w_mal ** 2)

    sinr = p_signal / (p_interference + p_noise)
    return {
        "P_signal_raw": p_signal_raw,
        "P_signal": p_signal,
        "P_interference": p_interference,
        "P_noise": p_noise,
        "SINR": sinr,
        "SINR_dB": 10 * np.log10(sinr),
    }


def compute_snr_closed_form(latest_flat, noise_std):
    """Scalar closed-form approximation of compute_snr() — see compute_sinr_closed_form()."""
    d = latest_flat[0].numel()
    M = len(latest_flat)
    powers = [(l.norm() ** 2 / d).item() for l in latest_flat]
    w = np.sqrt(np.mean(powers))

    p_noise = M * (noise_std ** 2)
    p_signal_raw = (M * w) ** 2
    p_signal = max(p_signal_raw - p_noise, 0.0)

    snr = p_signal / p_noise
    return {
        "P_signal_raw": p_signal_raw,
        "P_signal": p_signal,
        "P_noise": p_noise,
        "SNR": snr,
        "SNR_dB": 10 * np.log10(snr),
    }


def print_table(records, fmt="text", verbose=False):
    if verbose:
        cols = ["run", "variant", "malicious", "noise_std", "SINR_Psig", "SINR_Pinterf", "SINR_Pnoise", "SINR_dB",
                "SNR_Psig", "SNR_Pnoise", "SNR_dB"]
    else:
        cols = ["run", "variant", "malicious", "SINR_dB", "SNR_dB"]

    rows = []
    for r in records:
        row = {
            "run": r["run"],
            "variant": r["variant"],
            "malicious": ",".join(str(i) for i in r["malicious_idx"]),
            "SINR_dB": f'{r["SINR_dB"]:.2f}',
            "SNR_dB": f'{r["SNR_dB"]:.2f}',
        }
        if verbose:
            row.update({
                "noise_std": r["noise_std"],
                "SINR_Psig": f'{r["SINR_Psignal"]:.4f}',
                "SINR_Pinterf": f'{r["SINR_Pinterference"]:.4f}',
                "SINR_Pnoise": f'{r["SINR_Pnoise"]:.6f}',
                "SNR_Psig": f'{r["SNR_Psignal"]:.4f}',
                "SNR_Pnoise": f'{r["SNR_Pnoise"]:.6f}',
            })
        rows.append(row)

    if fmt == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    elif fmt == "markdown":
        print("| " + " | ".join(cols) + " |")
        print("|" + "|".join(["---"] * len(cols)) + "|")
        for row in rows:
            print("| " + " | ".join(str(row[c]) for c in cols) + " |")

    else:  # text
        widths = {"run": 10, "variant": 14, "malicious": 10}
        header = "".join(f'{c:<{widths.get(c, 11)}}' if c in ("run", "variant", "malicious")
                          else f'{c:>{widths.get(c, 11)}}' for c in cols)
        print(header)
        print("-" * len(header))
        for row in rows:
            print("".join(f'{row[c]:<{widths.get(c, 11)}}' if c in ("run", "variant", "malicious")
                           else f'{row[c]:>{widths.get(c, 11)}}' for c in cols))


def plot_results(records, out_path, title="SINR (pre-defense) vs SNR (post-defense) across runs"):
    """records: list of dicts with keys run, variant, n_users, SINR_dB, SNR_dB."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = sorted({r["n_users"] for r in records})
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.2), sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, n_users in zip(axes, panels):
        panel_records = [r for r in records if r["n_users"] == n_users]
        runs = sorted({r["run"] for r in panel_records})
        x = np.arange(len(runs))

        for variant in ("2-malicious", "1-malicious"):
            for metric in ("SINR", "SNR"):
                y = []
                for run in runs:
                    match = [r for r in panel_records if r["run"] == run and r["variant"] == variant]
                    y.append(match[0][f"{metric}_dB"] if match else np.nan)
                ax.plot(
                    x, y,
                    color=SERIES_COLORS[(metric, variant)],
                    linewidth=2.2,
                    marker="o",
                    markersize=7,
                    markeredgecolor="white",
                    markeredgewidth=1.4,
                    zorder=3,
                )

        ax.set_title(f"{n_users} users", color="#0b0b0b", fontsize=12, fontweight="bold", loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels(runs, color=TEXT_SECONDARY)
        ax.set_xlabel("run (increasing noise std →)", color=TEXT_MUTED, fontsize=9)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.tick_params(axis="both", colors=TEXT_MUTED, length=0)
        ax.margins(x=0.15)

    axes[0].set_ylabel("dB", color=TEXT_SECONDARY, fontsize=10)

    legend_handles = [
        Line2D([0], [0], color=color, linewidth=2.2, marker="o", markersize=6,
               markeredgecolor="white", markeredgewidth=1.2,
               label=f"{metric} ({variant})")
        for (metric, variant), color in SERIES_COLORS.items()
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.06), fontsize=9, labelcolor=TEXT_SECONDARY,
    )
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run folders to process")
    parser.add_argument("--noise-std", type=str, default="0.01",
                         help="Std of the per-element noise applied to each user's weights before saving. "
                              "Either a single value applied to every run (default: 0.01), or a "
                              "comma-separated list matching the run folders positionally, e.g. "
                              "'0.01,0.03,0.05' for '001_8 003_8 005_8'.")
    parser.add_argument("--malicious-idx", type=str, default=None,
                         help="Comma-separated 0-indexed malicious user indices. Only applied to a variant "
                              "whose (first-latest) count matches the number of indices given; otherwise "
                              "that variant falls back to auto-detection.")
    parser.add_argument("--gains", type=str, default=None,
                         help="Per-user channel gain |h_k| for the SINR (first_w_locals) computation: "
                              "comma-separated list or path to a .npy file, length = total users in that "
                              "variant. Defaults to unit gain. Not applied to the SNR (latest_w_locals) "
                              "computation, which always uses unit gain.")
    parser.add_argument("--verbose", action="store_true", help="Show P_signal / P_interference / P_noise breakdown")
    parser.add_argument("--format", choices=["text", "markdown", "csv"], default="text",
                         help="Table output format (default: text)")
    parser.add_argument("--plot", type=str, default=None,
                         help="Save a SINR/SNR-vs-run PNG to this path, faceted by user count (requires matplotlib)")
    parser.add_argument("--closed-form", action="store_true",
                         help="Use the scalar closed-form approximation (README 'Closed-form scalar "
                              "approximation') instead of the exact per-vector computation")
    args = parser.parse_args()

    if "," in args.noise_std:
        noise_std_list = [float(x) for x in args.noise_std.split(",")]
        if len(noise_std_list) != len(args.runs):
            parser.error(f"--noise-std has {len(noise_std_list)} values but {len(args.runs)} run folders were given")
        noise_std_by_run = dict(zip(args.runs, noise_std_list))
    else:
        noise_std_by_run = {run: float(args.noise_std) for run in args.runs}

    forced_malicious = None
    if args.malicious_idx is not None:
        forced_malicious = sorted(int(x) for x in args.malicious_idx.split(","))

    records = []
    for run in args.runs:
        noise_std = noise_std_by_run[run]
        for label, suffix in VARIANTS:
            loaded = load_locals(run, suffix)
            if loaded is None:
                continue
            first_flat, latest_flat = loaded

            n_malicious = len(first_flat) - len(latest_flat)
            if forced_malicious is not None and len(forced_malicious) == n_malicious:
                malicious_idx = forced_malicious
            else:
                malicious_idx = detect_malicious(first_flat, latest_flat)

            gains = load_gains(args.gains, len(first_flat))
            if args.closed_form:
                sinr_res = compute_sinr_closed_form(first_flat, malicious_idx, noise_std, gains)
                snr_res = compute_snr_closed_form(latest_flat, noise_std)
            else:
                sinr_res = compute_sinr(first_flat, malicious_idx, noise_std, gains)
                snr_res = compute_snr(latest_flat, noise_std)

            records.append({
                "run": run,
                "variant": label,
                "n_users": len(first_flat),
                "malicious_idx": malicious_idx,
                "noise_std": noise_std,
                "SINR_dB": sinr_res["SINR_dB"],
                "SINR_Psignal": sinr_res["P_signal"],
                "SINR_Pinterference": sinr_res["P_interference"],
                "SINR_Pnoise": sinr_res["P_noise"],
                "SNR_dB": snr_res["SNR_dB"],
                "SNR_Psignal": snr_res["P_signal"],
                "SNR_Pnoise": snr_res["P_noise"],
            })

    print_table(records, fmt=args.format, verbose=args.verbose)

    if args.plot:
        if not records:
            print("\nNo data loaded — nothing to plot.")
        else:
            title = "SINR (pre-defense) vs SNR (post-defense) across runs"
            if args.closed_form:
                title += " — closed-form approximation"
            plot_results(records, args.plot, title=title)
            print(f"\nSaved plot to {args.plot}")


if __name__ == "__main__":
    main()
