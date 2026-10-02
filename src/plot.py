import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # draw to a file; works on Colab and machines without a screen
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

# Colors follow the algorithm, so PPO is always blue and DQN always orange across charts
ALGO_COLORS = {"ppo": "#2a78d6", "dqn": "#eb6834"}
EXTRA_COLORS = ["#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

FRAMES_PER_STEP = 4  # the agent acts once every 4 game frames (see atari.py)


def load_run(run_dir):
    """Every finished game in a run, in the order played, with the training step it ended at."""
    games = []
    for csv in sorted((run_dir / "monitor").glob("*.monitor.csv")):
        df = pd.read_csv(csv, skiprows=1)  # first line is a JSON header
        # t is seconds since the run (or the latest resume) started; it restarts at 0 after
        # --resume, so add the time already elapsed to keep it increasing
        restarts = (df["t"].diff() < 0).cumsum()
        offsets = df.groupby(restarts)["t"].max().cumsum().shift(fill_value=0)
        df["t"] = df["t"] + restarts.map(offsets)
        games.append(df)
    if not games:
        raise SystemExit(f"No monitor/*.monitor.csv files in {run_dir}; is it a train.py run folder?")

    df = pd.concat(games).sort_values("t", ignore_index=True)
    # Monitor counts game frames (l) across all parallel games; divide to get agent steps
    df["step"] = df["l"].cumsum() / FRAMES_PER_STEP
    return df


def steps_label(x, _pos=None):
    if x >= 999_500:  # anything that would round to "1000k" reads as 1M
        return f"{x / 1e6:.3g}M"
    if x >= 1e3:
        return f"{x / 1e3:.3g}k"
    return f"{x:.0f}"


def main():
    parser = argparse.ArgumentParser(description="Plot learning curves (score per game vs training steps)")
    parser.add_argument("runs", nargs="+", help="train.py run folders, e.g. runs/ppo_seed0 runs/dqn_seed0")
    parser.add_argument("--random-eval", help="random baseline JSON from evaluate.py (default: random_eval.json beside the runs)")
    parser.add_argument("--window", type=int, default=100, help="games averaged into each point of the curve")
    parser.add_argument("--save", help="output image (default: learning_curves.png beside the runs)")
    args = parser.parse_args()

    run_dirs = [Path(r) for r in args.runs]
    parent = run_dirs[0].parent
    save = Path(args.save or parent / "learning_curves.png")
    random_eval = Path(args.random_eval or parent / "random_eval.json")

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    spare = iter(EXTRA_COLORS)
    used = set()
    summary = []
    for run_dir in run_dirs:
        df = load_run(run_dir)
        name = run_dir.name
        algo = name.split("_")[0]
        color = ALGO_COLORS.get(algo) if algo not in used else None
        color = color or next(spare)
        used.add(algo)

        smooth = df["r"].rolling(args.window, min_periods=min(10, len(df))).mean()
        label = name.replace("_seed", " · seed ").upper().replace("SEED", "seed")

        ax.scatter(df["step"], df["r"], s=6, color=color, alpha=0.3, linewidths=0, zorder=1)
        ax.plot(df["step"], smooth, color=color, linewidth=2, label=label, zorder=3)
        last = smooth.dropna()
        if not last.empty:
            ax.annotate(f"{algo.upper()} {last.iloc[-1]:.1f}", (df["step"].iloc[-1], last.iloc[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center", fontsize=9, color=INK, zorder=4)

        recent = df["r"].tail(args.window)
        summary.append((name, df["step"].iloc[-1], df["t"].iloc[-1] / 3600, len(df), recent.mean(), recent.std()))

    if random_eval.exists():
        baseline = json.loads(random_eval.read_text())["mean"]
        ax.axhline(baseline, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2, label="Random agent")
        ax.annotate(f"random {baseline:.1f}", (0, baseline), xytext=(4, 4), textcoords="offset points",
                    fontsize=9, color=INK_2, va="bottom")

    ax.set_title(f"Pac-Man score while training (each line averages the last {args.window} games)",
                 loc="left", fontsize=12, color=INK, pad=12)
    ax.set_xlabel("Training steps", color=INK_2, fontsize=10)
    ax.set_ylabel("Score per game", color=INK_2, fontsize=10)
    ax.xaxis.set_major_formatter(FuncFormatter(steps_label))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK_2)
    # Room on the right for the end-of-line labels
    ax.margins(x=0.08)

    save.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save, facecolor=fig.get_facecolor())

    print(f"{'run':<16}{'steps':>10}{'hours':>8}{'games':>8}   score (last {args.window} games)")
    for name, steps, hours, n, mean, std in summary:
        print(f"{name:<16}{steps_label(steps):>10}{hours:>8.2f}{n:>8}   {mean:.1f} ± {std:.1f}")
    print(f"Plot saved to {save}")


if __name__ == "__main__":
    main()
