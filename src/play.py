import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

# 1. Use the Atari wrappers to match training, but set render_mode to human
env = make_atari_env("ALE/Pacman-v5", n_envs=1, env_kwargs={"render_mode": "human"})

# 2. Stack 4 frames
env = VecFrameStack(env, n_stack=4)

# 3. Load the CNN model
model = PPO.load("models/pacman_ppo_cnn")
print("CNN Model loaded! Watch it play...")

obs = env.reset()

# Let it play for a few minutes
for _ in range(5000):
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)

env.close()