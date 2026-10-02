import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from atari import ENV_ID, make_env

ALGOS = {"ppo": PPO, "dqn": DQN}


class RandomAgent:
    """Presses random buttons; the baseline a trained agent has to beat."""

    def __init__(self, env):
        self.env = env

    def predict(self, obs, state=None, episode_start=None, deterministic=False):
        return np.array([self.env.action_space.sample() for _ in range(self.env.num_envs)]), None


def main():
    parser = argparse.ArgumentParser(description=f"Score an agent over full games of {ENV_ID}")
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--model", help="path to a saved model .zip")
    who.add_argument("--random", action="store_true", help="evaluate random button presses instead")
    parser.add_argument("--algo", choices=ALGOS, default="ppo")
    parser.add_argument("--episodes", type=int, default=100, help="number of full games to play")
    parser.add_argument("--n-envs", type=int, default=8, help="games played in parallel (only affects speed)")
    parser.add_argument("--seed", type=int, default=1000, help="kept apart from training seeds so games are unseen")
    parser.add_argument("--stochastic", action="store_true", help="sample moves instead of always picking the best one")
    parser.add_argument("--save", help="where to write the results JSON (default: next to the model, or runs/random_eval.json)")
    args = parser.parse_args()

    env = make_env(n_envs=args.n_envs, seed=args.seed)
    env.action_space.seed(args.seed)

    if args.random:
        agent, label = RandomAgent(env), "random"
        save = Path(args.save or "runs/random_eval.json")
    else:
        agent, label = ALGOS[args.algo].load(args.model, device="auto"), args.model
        save = Path(args.save or Path(args.model).with_suffix("").as_posix() + "_eval.json")

    print(f"Playing {args.episodes} full games of {ENV_ID} with {label}...")
    # The Monitor wrapper reports each full game (all lives) with its real, unclipped Pac-Man score
    scores, lengths = evaluate_policy(
        agent, env, n_eval_episodes=args.episodes, deterministic=not args.stochastic, return_episode_rewards=True
    )
    env.close()

    scores = np.array(scores)
    results = {
        "env_id": ENV_ID,
        "agent": "random" if args.random else args.algo,
        "model": args.model,
        "trained_steps": None if args.random else int(agent.num_timesteps),
        "episodes": len(scores),
        "deterministic": None if args.random else not args.stochastic,
        "seed": args.seed,
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "median": float(np.median(scores)),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "scores": scores.tolist(),
        "lengths_frames": [int(n) for n in lengths],
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save.parent.mkdir(parents=True, exist_ok=True)
    save.write_text(json.dumps(results, indent=2))

    print(f"Score over {len(scores)} games: {results['mean']:.1f} ± {results['std']:.1f} "
          f"(median {results['median']:.0f}, min {results['min']:.0f}, max {results['max']:.0f})")
    print(f"Results saved to {save}")


if __name__ == "__main__":
    main()
