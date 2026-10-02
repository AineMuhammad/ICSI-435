"""Shared Atari setup, so training and playing see the game exactly the same way."""
import gymnasium as gym
import ale_py
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

ENV_ID = "ALE/Pacman-v5"


def make_env(env_id=ENV_ID, n_envs=1, seed=0, render_mode=None, monitor_dir=None, append_monitor=False):
    # v5 games already skip 4 frames per action, and SB3's Atari wrapper skips 4 more.
    # frameskip=1 turns off the game's own skip so the agent acts every 4 frames, not 16.
    env_kwargs = {"frameskip": 1}
    if render_mode:
        env_kwargs["render_mode"] = render_mode

    # Atari wrappers: grayscale, resize to 84x84, clip rewards, end episode on life lost.
    # Monitor logs each episode's real (unclipped) score to monitor_dir for plotting later.
    env = make_atari_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
        monitor_dir=monitor_dir,
        monitor_kwargs={"override_existing": not append_monitor},
    )

    # Stack 4 frames together so the AI can perceive motion (velocity)
    return VecFrameStack(env, n_stack=4)
