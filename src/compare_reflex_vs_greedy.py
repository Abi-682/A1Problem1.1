import sys
from typing import Dict, Any

# Ensure src is on path when running from repo root
sys.path.append("src")

import random
import matplotlib.pyplot as plt

from warehouse_env import WarehouseEnv
from warehouse_agent_reflex import WarehouseAgentReflex
from warehouse_agent_greedy import GreedyManhattanAgent
from run_episode import run_episode


def run_episodes(agent_factory, N: int = 50) -> Dict[str, Any]:
    successes = []
    episode_lengths = []
    final_batteries = []
    total_rewards = []

    for i in range(N):
        env = WarehouseEnv()
        agent = agent_factory(env)
        # run without recording frames for speed
        res = run_episode(env, agent, randomize=True, record_frames=False, max_steps=500)
        successes.append(bool(res.get("terminated", False)))
        episode_lengths.append(int(res.get("steps", 0)))
        final_batteries.append(int(res.get("battery", 0)))
        total_rewards.append(float(res.get("total_reward", 0.0)))

    return {
        "success_rate": float(sum(1 for s in successes if s)) / max(1, N),
        "successes": successes,
        "episode_lengths": episode_lengths,
        "final_batteries": final_batteries,
        "total_rewards": total_rewards,
    }


def plot_comparison(stats_a, stats_b, labels=("Reflex", "Greedy"), out_path="comparison.png"):
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    axs[0].bar([0, 1], [stats_a["success_rate"], stats_b["success_rate"]], color=["C0", "C1"])
    axs[0].set_xticks([0, 1])
    axs[0].set_xticklabels(labels)
    axs[0].set_ylim(0, 1)
    axs[0].set_title("Success Rate")

    axs[1].boxplot([stats_a["episode_lengths"], stats_b["episode_lengths"]], labels=labels, notch=True)
    axs[1].set_title("Episode Lengths (steps)")

    axs[2].hist(stats_a["final_batteries"], bins=10, alpha=0.6, label=labels[0])
    axs[2].hist(stats_b["final_batteries"], bins=10, alpha=0.6, label=labels[1])
    axs[2].set_title("Final Battery Levels")
    axs[2].legend()

    plt.tight_layout()
    fig.savefig(out_path)
    print(f"Saved comparison plot to {out_path}")


def main():
    N = 50
    print(f"Running {N} episodes per agent...")

    stats_reflex = run_episodes(lambda env: WarehouseAgentReflex(), N=N)
    stats_greedy = run_episodes(lambda env: GreedyManhattanAgent(env), N=N)

    print("Reflex success rate:", stats_reflex["success_rate"]) 
    print("Greedy success rate:", stats_greedy["success_rate"]) 

    plot_comparison(stats_reflex, stats_greedy, labels=("Reflex", "Greedy"), out_path="comparison.png")


if __name__ == "__main__":
    main()
