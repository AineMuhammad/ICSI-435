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

To watch a trained agent locally (needs a screen, so not on Colab):

```
python src/play.py --algo ppo --model runs/ppo_seed0/final_model.zip
```
