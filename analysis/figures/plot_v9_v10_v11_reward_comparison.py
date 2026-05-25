from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "analysis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOGS = {
    "v9": [Path("D:/train_graphcredit_math_v9_nonsharing_solver_verifier.log")],
    "v10": [
        Path("D:/train_graphcredit_math_v10_nonsharing_solver_verifier.log"),
        Path("D:/train_graphcredit_math_v10_nonsharing_solver_verifier_resume200.log"),
    ],
    "v11": [Path("D:/train_graphcredit_math_v11_nonsharing_solver_verifier.log")],
}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(\d+) - (.*)")

METRICS = [
    "graphcredit/node_reward_mean",
    "graphcredit/solver_node_reward_mean",
    "graphcredit/verifier_node_reward_mean",
    "graphcredit/verifier_negative_node_ratio",
]


def parse_log(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = ANSI_RE.sub("", line)
            match = STEP_RE.search(line)
            if not match:
                continue
            row: dict[str, float] = {"step": float(match.group(1))}
            for part in match.group(2).split(" - "):
                if ":" not in part:
                    continue
                key, value = part.rsplit(":", 1)
                key = key.strip()
                try:
                    row[key] = float(value.strip())
                except ValueError:
                    pass
            if any(metric in row for metric in METRICS):
                rows.append(row)
    return rows


def load_runs() -> dict[str, list[dict[str, float]]]:
    runs: dict[str, list[dict[str, float]]] = {}
    for name, paths in LOGS.items():
        by_step: dict[int, dict[str, float]] = {}
        for path in paths:
            for row in parse_log(path):
                by_step[int(row["step"])] = row
        runs[name] = [by_step[step] for step in sorted(by_step)]
    return runs


def rolling_mean(rows: list[dict[str, float]], metric: str, window: int = 25) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for idx, row in enumerate(rows):
        start = max(0, idx - window + 1)
        values = [item[metric] for item in rows[start : idx + 1] if metric in item]
        if not values or metric not in row:
            continue
        xs.append(int(row["step"]))
        ys.append(sum(values) / len(values))
    return xs, ys


def save_csv(runs: dict[str, list[dict[str, float]]]) -> None:
    csv_path = OUT_DIR / "v9_v10_v11_reward_comparison.csv"
    fieldnames = ["version", "step", *METRICS, "episode/pass@2", "episode/avg@2", "timing_s/testing"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for version, rows in runs.items():
            for row in rows:
                writer.writerow({"version": version, **{key: row.get(key, "") for key in fieldnames[1:]}})


def plot(runs: dict[str, list[dict[str, float]]]) -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )
    colors = {"v9": "#4C78A8", "v10": "#F58518", "v11": "#54A24B"}
    panels = [
        ("graphcredit/node_reward_mean", "Overall node reward"),
        ("graphcredit/solver_node_reward_mean", "Solver node reward"),
        ("graphcredit/verifier_node_reward_mean", "Verifier node reward"),
        ("graphcredit/verifier_negative_node_ratio", "Verifier negative ratio"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), sharex=True)
    for ax, (metric, ylabel) in zip(axes.ravel(), panels, strict=True):
        for version, rows in runs.items():
            xs, ys = rolling_mean(rows, metric)
            if not xs:
                continue
            ax.plot(xs, ys, label=version, color=colors[version], linewidth=2.0)
        ax.set_ylabel(ylabel)
        if metric == "graphcredit/verifier_negative_node_ratio":
            ax.set_ylim(-0.02, 0.75)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    for ax in axes[-1]:
        ax.set_xlabel("Training step")
    axes[0, 0].legend(loc="best", ncol=3)
    fig.tight_layout()
    png_path = OUT_DIR / "v9_v10_v11_reward_comparison.png"
    pdf_path = OUT_DIR / "v9_v10_v11_reward_comparison.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)


def main() -> None:
    runs = load_runs()
    save_csv(runs)
    plot(runs)


if __name__ == "__main__":
    main()
