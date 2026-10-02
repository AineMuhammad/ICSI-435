# AI in Retro Games (ICSI 435/535, Group 13)

Training a reinforcement learning agent to play Atari Pac-Man (`ALE/Pacman-v5`)
with Stable-Baselines3.

Run every command from the repository root, so `models/` and `logs/` end up there.

## Option A: Google Colab (recommended, free GPU)

1. Go to https://colab.research.google.com and choose **New notebook**.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Run this in the first cell (the `!` and `%` run shell commands from a notebook):

   ```
   !git clone -b dev https://github.com/EricMaizner/ICSI-435.git
   %cd ICSI-435
   !pip install -r requirements.txt
   ```

4. Smoke test (see [Run the test program](#run-the-test-program)):

   ```
   !python src/test.py
   !python src/test_play.py --headless
   ```

Colab wipes its files when the session ends. To keep trained models, mount Google Drive
(`from google.colab import drive; drive.mount('/content/drive')`) and copy `models/` there.

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
