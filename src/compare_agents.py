"""
compare_agents.py

Run multiple episodes for agents, collect statistics, and visualize comparisons.

Deliverables:
- run_episodes(agent_factory, N): runs N episodes and returns stats dict
- compare_and_plot(stats_a, stats_b, labels): creates figure with 3 subplots

"""
from typing import Callable, Dict, Any, List
import random

from warehouse_env import WarehouseEnv
from warehouse_agent_reflex import ReflexAgent


def run_episodes(agent_factory: Callable[[WarehouseEnv], Any], N: int = 100, randomize: bool = True, max_steps: int = 500) -> Dict[str, Any]:
    """
    Run N episodes of an agent and collect statistics.

    agent_factory: callable that accepts an env and returns an agent instance (agent(env)).
    Returns a dictionary with:
      - 'success_rate': float
      - 'successes': list[bool]
      - 'episode_lengths': list[int]
      - 'final_batteries': list[int]
      - 'total_rewards': list[float]
    """
    successes: List[bool] = []
    episode_lengths: List[int] = []
    final_batteries: List[int] = []
    total_rewards: List[float] = []

    for i in range(N):
        env = WarehouseEnv()
        agent = agent_factory(env)
        obs = env.reset(randomize=randomize)

        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0

        while steps < max_steps:
            action = agent.decide(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break

        successes.append(bool(terminated))
        episode_lengths.append(steps)
        final_batteries.append(obs["battery"])
        total_rewards.append(total_reward)

    success_rate = float(sum(1 for s in successes if s)) / max(1, N)
    return {
        "success_rate": success_rate,
        "successes": successes,
        "episode_lengths": episode_lengths,
        "final_batteries": final_batteries,
        "total_rewards": total_rewards,
    }


class RandomAgent:
    """Simple baseline agent that picks random valid moves."""

    def __init__(self, env: WarehouseEnv):
        self.env = env
        self.actions = env.ACTIONS

    def decide(self, observation: Dict[str, object]):
        local = observation.get("local_grid")
        # Prefer any non-wall movement when possible
        if local:
            view_radius = len(local) // 2
            center = view_radius
            moves = []
            if local[center - 1][center] != "#":
                moves.append("N")
            if local[center + 1][center] != "#":
                moves.append("S")
            if local[center][center - 1] != "#":
                moves.append("W")
            if local[center][center + 1] != "#":
                moves.append("E")
            if moves:
                return random.choice(moves)
        # Fallback to any action
        return random.choice(self.actions)


def compare_and_plot(stats_a: Dict[str, Any], stats_b: Dict[str, Any], labels=("Agent A", "Agent B"), save_path: str | None = None) -> None:
    """
    Create a figure with 3 subplots:
      1) Bar chart of success rates
      2) Box plots of episode lengths
      3) Histograms of final battery levels
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; cannot plot results.")
        return

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))

    # Subplot 1: Success rates (bar)
    axs[0].bar([0, 1], [stats_a["success_rate"], stats_b["success_rate"]], color=["C0", "C1"])
    axs[0].set_xticks([0, 1])
    axs[0].set_xticklabels(labels)
    axs[0].set_ylim(0, 1)
    axs[0].set_title("Success Rate")
    axs[0].set_ylabel("Fraction of successful episodes")

    # Subplot 2: Episode lengths (boxplot)
    axs[1].boxplot([stats_a["episode_lengths"], stats_b["episode_lengths"]], labels=labels, notch=True)
    axs[1].set_title("Episode Lengths (steps)")
    axs[1].set_ylabel("Steps")

    # Subplot 3: Final battery histograms
    axs[2].hist(stats_a["final_batteries"], bins=10, alpha=0.6, label=labels[0])
    axs[2].hist(stats_b["final_batteries"], bins=10, alpha=0.6, label=labels[1])
    axs[2].set_title("Final Battery Levels")
    axs[2].set_xlabel("Battery")
    axs[2].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    # Quick comparison demo: ReflexAgent vs RandomAgent
    N = 50
    print(f"Running {N} episodes per agent (randomized start/pickup/dropoff)...")

    stats_reflex = run_episodes(lambda env: ReflexAgent(env), N=N, randomize=True)
    stats_random = run_episodes(lambda env: RandomAgent(env), N=N, randomize=True)

    print("Reflex Agent success rate:", stats_reflex["success_rate"]) 
    print("Random Agent success rate:", stats_random["success_rate"]) 

    compare_and_plot(stats_reflex, stats_random, labels=("Reflex", "Random"), save_path=None)
