# AI in Retro Games (ICSI 435/535, Group 13)

Training a reinforcement learning agent to play Atari Pac-Man (`ALE/Pacman-v5`)
with Stable-Baselines3.

Run every command from the repository root, so output folders (`runs/`, `models/`, `logs/`) end up there.

## Option A: Google Colab (recommended, free GPU)

1. Go to https://colab.research.google.com and choose **New notebook**.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Run this in the first cell (the `!` and `%` run shell commands from a notebook):

   ```
   !git clone -b dev https://github.com/AineMuhammad/ICSI-435.git
   %cd ICSI-435
   !pip install -r requirements.txt
   ```

4. Smoke test (see [Run the test program](#run-the-test-program)):

   ```
   !python src/test.py
   !python src/test_play.py --headless
   ```

Colab wipes its files when the session ends, so save training runs to Google Drive
(see [Train on Pac-Man](#train-on-pac-man)). After new changes are pushed, run `!git pull`
inside `ICSI-435` instead of cloning again.

## Option B: Local machine

(While Workspace is open, open a new terminal)

```
# Create the environment
python -m venv rl-env

# Activate it (Windows)
.\rl-env\Scripts\activate

# Activate it (Mac/Linux)
source rl-env/bin/activate

# Install Stable-Baselines3, Gymnasium and the Atari emulator (includes the game ROMs)
pip install -r requirements.txt
```

Once everything is installed, your terminal lines should begin with "(rl-env)".

## Run the test program

`src/test.py` trains a quick CartPole model (a pole-balancing task) to check the setup works.
Wait until training stops, then run `src/test_play.py` (add `--headless` on Colab, where no
game window can open). It plays 5 games; the results should be mixed, with some low scores
and some perfect 500s.

## Train on Pac-Man

`src/train.py` trains a PPO or DQN agent on `ALE/Pacman-v5`. Each run goes in its own folder,
`<out>/<algo>_seed<seed>/`:

| Path | Contents |
|---|---|
| `checkpoints/model_<steps>_steps.zip` | Saved every `--checkpoint-freq` steps (default 100,000) |
| `final_model.zip` | Model at the end of training |
| `monitor/*.monitor.csv` | Real game score and length of every episode |
| `tb/` | TensorBoard logs |

| Option | Default | Meaning |
|---|---|---|
| `--algo` | `ppo` | `ppo` or `dqn` |
| `--timesteps` | 1,000,000 | Total training steps |
| `--seed` | 0 | Random seed; a different seed gives a separate run folder |
| `--n-envs` | 8 (PPO), 1 (DQN) | Games played in parallel |
| `--out` | `runs` | Where run folders are created |
| `--resume` | off | Continue from the latest checkpoint up to `--timesteps` |

On Colab, mount Google Drive first so runs survive disconnects:

```
from google.colab import drive
drive.mount('/content/drive')
```

```
!python src/train.py --algo ppo --timesteps 20000 --checkpoint-freq 10000 --out /content/drive/MyDrive/ICSI-435/runs
```

If the session drops, reconnect, re-run the setup cell and the Drive cell, then add `--resume`
to the same command. DQN's replay buffer is not saved, so after resuming it refills from scratch.
Keep `--timesteps` the same when resuming: PPO's learning rate and DQN's exploration decay over
`--timesteps`, so raising it mid-run restarts part of that decay.

To watch a trained agent locally (needs a screen, so not on Colab):

```
python src/play.py --algo ppo --model runs/ppo_seed0/final_model.zip
```

## Evaluate an agent

`src/evaluate.py` plays full games (all lives) and reports the real Pac-Man score as
mean ± std, median, min and max. Results are saved as JSON next to the model
(`final_model_eval.json`) or wherever `--save` points.

```
# Baseline: random button presses (the bar a trained agent has to beat)
!python src/evaluate.py --random --save /content/drive/MyDrive/ICSI-435/runs/random_eval.json

# A trained model
!python src/evaluate.py --algo ppo --model /content/drive/MyDrive/ICSI-435/runs/ppo_seed0/final_model.zip
```

| Option | Default | Meaning |
|---|---|---|
| `--episodes` | 100 | Full games to play |
| `--seed` | 1000 | Kept apart from training seeds so the games are unseen |
| `--stochastic` | off | Sample moves instead of always picking the best one |
| `--n-envs` | 8 | Games played in parallel (only affects speed) |

By default the agent always picks its best move. A barely trained agent can get stuck doing the
same move every step (e.g. always LEFT into a wall); `--stochastic` shows how it scores when it
samples moves the way it does during training.

## Compare algorithms (learning curves)

`src/plot.py` reads each run's `monitor/` logs and draws score per game against training steps,
one line per run (PPO blue, DQN orange), averaged over the last `--window` games (default 100),
with the random baseline as a dashed line if `random_eval.json` sits beside the runs. It also
prints a summary table (steps, hours, games, recent score) for each run.

```
!python src/plot.py $RUNS/ppo_seed0 $RUNS/dqn_seed0
```

The image is saved as `learning_curves.png` beside the runs (or at `--save`). Steps are
estimated from finished games (4 frames per step), so they read slightly below the true count;
games still in progress when training stopped are not counted.

### The DQN vs. PPO comparison (slide 6)

Train both for the same budget, then evaluate and plot (about 35 min for PPO and 65 min for DQN
on a Colab T4; keep the tab open):

```
!python src/train.py --algo ppo --timesteps 1000000 --out $RUNS
!python src/train.py --algo dqn --timesteps 1000000 --out $RUNS
!python src/evaluate.py --algo ppo --model $RUNS/ppo_seed0/final_model.zip
!python src/evaluate.py --algo dqn --model $RUNS/dqn_seed0/final_model.zip
!python src/plot.py $RUNS/ppo_seed0 $RUNS/dqn_seed0
```

`$RUNS` is the Drive folder set in the Colab setup (`RUNS = "/content/drive/MyDrive/ICSI-435/runs"`).
If either run already exists from a test, delete its folder first or pass a new `--seed`.
