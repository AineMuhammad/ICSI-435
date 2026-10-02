import gymnasium as gym
from stable_baselines3 import PPO

# 1. Create the CartPole environment
env = gym.make("CartPole-v1")

# 2. Use MlpPolicy for simple numerical data (no vision wrappers needed)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/")

# 3. Train for 10,000 timesteps
print("Starting CartPole training...")
model.learn(total_timesteps=10000)

# 4. Save the trained model inside the project's models folder
model.save("models/cartpole_model")

env.close()
print("Training complete. Model saved to models/cartpole_model.zip")