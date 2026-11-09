#!/usr/bin/env python3
"""Reproducible *random policy* baseline for the legacy four-tuple TetrisEnv.

The numbers measure only this environment/action policy. They are not DQN
training results or comparable against different episodes/seed budgets.
"""
import argparse
import json
import random
import statistics

from tetris_env import TetrisEnv


def run(seeds=(2026, 2027, 2028), episodes=3, max_steps=400):
    if episodes < 1 or max_steps < 1:
        raise ValueError("episodes and max_steps must be positive")
    env = TetrisEnv()
    records = []
    try:
        for seed in seeds:
            for episode in range(episodes):
                episode_seed = int(seed) * 1000 + episode
                env.reset(seed=episode_seed)
                env.max_steps = max_steps
                policy_rng = random.Random(episode_seed + 8491)
                reward_total = 0.0
                done = False
                while not done:
                    _, reward, done, info = env.step(policy_rng.randrange(env.action_space.n))
                    reward_total += float(reward)
                records.append({
                    "seed": int(seed), "episode": episode,
                    "score": int(info["score"]), "lines": int(info["lines"]),
                    "steps": int(info["steps"]),
                    "reward": round(reward_total, 6),
                    "truncated": bool(info["truncated"]),
                })
    finally:
        env.close()
    return {
        "algorithm": "uniform-random-actions",
        "environment": "TetrisEnv legacy Gym four-tuple",
        "seeds": list(seeds),
        "episodes_per_seed": episodes,
        "max_steps": max_steps,
        "mean_score": round(statistics.mean(r["score"] for r in records), 6),
        "mean_lines": round(statistics.mean(r["lines"] for r in records), 6),
        "results": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    print(json.dumps(run(episodes=args.episodes, max_steps=args.steps),
                     sort_keys=True, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
