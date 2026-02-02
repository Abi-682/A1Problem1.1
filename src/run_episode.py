"""Run a single episode runner usable by scripts and notebooks.

Provides `run_episode(env, agent, ...)` which returns a result dict and
optionally records frames and metrics for visualization.
Supports agent APIs with either `decide(observation)` or `act(state)`.
"""
from typing import Any, Dict

from warehouse_env import WarehouseEnv
from warehouse_viz import replay_animation


def _manhattan(a, b):
    if a is None or b is None:
        return 0
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def run_episode(env: WarehouseEnv, agent: Any, randomize: bool = True, record_frames: bool = True, max_steps: int = 500) -> Dict[str, object]:
    """Run one episode using provided env and agent.

    agent may implement `decide(observation)` or `act(state)`.
    Returns a dict with keys: terminated (bool), truncated (bool), steps, battery,
    total_reward, frames (if recorded), metrics (if recorded).
    """
    obs = env.reset(randomize=randomize)

    frames = []
    metrics = {"rewards": [], "battery": [], "dist_pickup": [], "dist_dropoff": []} if record_frames else None

    if record_frames:
        frames.append(env.render_grid())
        metrics["rewards"].append(0.0)
        metrics["battery"].append(obs["battery"])
        metrics["dist_pickup"].append(_manhattan(obs["robot_pos"], obs.get("pickup_pos")))
        metrics["dist_dropoff"].append(_manhattan(obs["robot_pos"], obs.get("dropoff_pos")))

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    # Helper to call agent
    def _agent_action(observation):
        if hasattr(agent, "decide"):
            return agent.decide(observation)
        if hasattr(agent, "act"):
            return agent.act(observation)
        raise RuntimeError("Agent must implement decide(obs) or act(state)")

    while steps < max_steps:
        action = _agent_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if record_frames:
            frames.append(env.render_grid())
            metrics["rewards"].append(reward)
            metrics["battery"].append(obs["battery"])
            metrics["dist_pickup"].append(_manhattan(obs["robot_pos"], obs.get("pickup_pos")))
            metrics["dist_dropoff"].append(_manhattan(obs["robot_pos"], obs.get("dropoff_pos")))

        if terminated or truncated:
            break

    result = {
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "steps": steps,
        "battery": obs["battery"],
        "total_reward": total_reward,
    }
    if record_frames:
        result["frames"] = frames
        result["metrics"] = metrics

        # Launch visualization
        try:
            print("Launching animation replay... Controls: SPACE, LEFT/RIGHT")
            replay_animation(frames, metrics=metrics, interval_ms=200)
        except Exception as e:
            print("Visualization failed:", e)

    return result


if __name__ == "__main__":
    # Simple self-test: run ReflexAgent and Greedy agent demos if available
    try:
        from warehouse_agent_reflex import ReflexAgent, WarehouseAgentReflex
        from warehouse_agent_greedy import GreedyManhattanAgent
    except Exception:
        print("Agents not available for demo. Please run from project root.")
    else:
        env = WarehouseEnv()
        print("Demo: ReflexAgent")
        a = ReflexAgent(env)
        run_episode(env, a, randomize=True, record_frames=True, max_steps=500)

