import argparse

from stable_baselines3 import DQN, PPO

from atari import make_env

parser = argparse.ArgumentParser(description="Watch a trained agent play (needs a screen, so not on Colab)")
parser.add_argument("--algo", choices=["ppo", "dqn"], default="ppo")
parser.add_argument("--model", default="runs/ppo_seed0/final_model.zip", help="path to a saved model .zip")
parser.add_argument("--steps", type=int, default=5000, help="how many steps to watch")
args = parser.parse_args()

# 1. Same game setup as training, but with a window to watch it in
env = make_env(n_envs=1, render_mode="human")

# 2. Load the CNN model
model = (PPO if args.algo == "ppo" else DQN).load(args.model)
print("CNN Model loaded! Watch it play...")

obs = env.reset()

# Let it play for a few minutes
for _ in range(args.steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)

env.close()
