import argparse
import gymnasium as gym
from stable_baselines3 import PPO

# Use --headless where there is no screen to open a window on (e.g. Google Colab)
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="play without opening a game window")
args = parser.parse_args()

# 1. Create the environment, with the visual window unless running headless
env = gym.make("CartPole-v1", render_mode=None if args.headless else "human")

# 2. Load the model we just trained
try:
    model = PPO.load("models/cartpole_model")
    print("Model loaded successfully. Starting evaluation...")
except FileNotFoundError:
    print("Error: Could not find models/cartpole_model.zip.")
    exit()

# 3. Play 5 full games to verify performance
for episode in range(5):
    obs, info = env.reset()
    done = False
    score = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward

        # Game ends if the pole falls (terminated) or hits 500 frames (truncated)
        done = terminated or truncated

    print(f"Game {episode + 1} finished with a score of: {score}")

env.close()
