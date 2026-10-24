"""Per-game training log (games.csv): the real score of every finished game and the step it ended at."""
import time

import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

COLUMNS = ["step", "score", "frames", "hours"]
FRAMES_PER_STEP = 4  # the agent acts once every 4 game frames (see atari.py)


def read_monitor(run_dir):
    """Older runs logged games only in monitor/*.monitor.csv, which has no step column; estimate it."""
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
        return pd.DataFrame(columns=COLUMNS)

    df = pd.concat(games).sort_values("t", ignore_index=True)
    # Monitor counts game frames (l) across all parallel games; divide to get agent steps.
    # This reads slightly low, since games still in progress are not counted.
    return pd.DataFrame({"step": df["l"].cumsum() / FRAMES_PER_STEP, "score": df["r"],
                         "frames": df["l"], "hours": df["t"] / 3600})


def read_games(run_dir):
    """Every finished game in a run, in the order played, from games.csv plus any older monitor logs."""
    path = run_dir / "games.csv"
    if not path.exists():
        return read_monitor(run_dir)
    games = pd.read_csv(path)
    # A run resumed from an older version of train.py has its early games only in monitor/
    older = read_monitor(run_dir)
    if not games.empty:
        older = older[older["step"] < games["step"].iloc[0]]
    return pd.concat([older, games], ignore_index=True)


class GameLog(BaseCallback):
    """Collects finished games and writes them to games.csv each time a checkpoint is saved.

    Writing only alongside checkpoints keeps the log in step with them: if the session dies,
    the games since the last checkpoint are lost, but --resume replays those steps anyway.
    The file is rewritten and closed each time, so it reaches Google Drive even if Colab
    disconnects (Drive may never upload a file that is still open).
    """

    def __init__(self, path, save_freq, start_step):
        super().__init__()
        self.path = path
        self.save_freq = save_freq
        self.rows = []
        if path.exists():
            # Drop games after the checkpoint we resume from; those steps will be played again
            self.rows = pd.read_csv(path).query("step <= @start_step").to_dict("records")
            self.hours_before = self.rows[-1]["hours"] if self.rows else 0.0
        else:
            # Count training time already logged by older monitor files, if any
            older = read_monitor(path.parent)
            self.hours_before = older["hours"].max() if not older.empty else 0.0

    def _on_training_start(self):
        self.start = time.time()

    def _on_step(self):
        # Monitor adds "episode" to a game's info when the whole game ends (all lives lost)
        for info in self.locals["infos"]:
            if "episode" in info:
                hours = self.hours_before + (time.time() - self.start) / 3600
                self.rows.append({"step": self.num_timesteps, "score": info["episode"]["r"],
                                  "frames": info["episode"]["l"], "hours": round(hours, 4)})
        if self.n_calls % self.save_freq == 0:
            self.write()
        return True

    def _on_training_end(self):
        self.write()

    def write(self):
        pd.DataFrame(self.rows, columns=COLUMNS).to_csv(self.path, index=False)
