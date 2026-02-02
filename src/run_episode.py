"""
Run a complete episode with the reflex agent and visualize the results.
"""
import math
from warehouse_env import WarehouseEnv
from warehouse_agent_reflex import ReflexAgent
from warehouse_viz import replay_animation


def manhattan_distance(pos1: tuple, pos2: tuple) -> int:
    """Calculate Manhattan distance between two positions."""
    if pos1 is None or pos2 is None:
        return 0
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def run_episode(randomize: bool = True):
    """
    Run a complete episode with the reflex agent.
    
    Args:
        randomize: If True, randomize pickup/dropoff locations and start position
    """
    # Initialize environment and agent
    env = WarehouseEnv()
    agent = ReflexAgent(env)
    
    # Reset environment with optional randomization
    obs = env.reset(randomize=randomize)
    
    # Collections for replay and metrics
    frames = []
    metrics = {
        "rewards": [],
        "battery": [],
        "dist_pickup": [],
        "dist_dropoff": [],
    }
    
    # Record initial state
    frames.append(env.render_grid())
    metrics["rewards"].append(0.0)
    metrics["battery"].append(obs["battery"])
    metrics["dist_pickup"].append(manhattan_distance(obs["robot_pos"], obs["pickup_pos"]))
    metrics["dist_dropoff"].append(manhattan_distance(obs["robot_pos"], obs["dropoff_pos"]))
    
    print("=" * 60)
    print("WAREHOUSE EPISODE RUNNER")
    print("=" * 60)
    print(f"Starting position: {obs['robot_pos']}")
    print(f"Pickup location: {obs['pickup_pos']}")
    print(f"Dropoff location: {obs['dropoff_pos']}")
    print("=" * 60)
    print()
    
    # Run episode
    total_reward = 0.0
    step_count = 0
    done = False
    
    while not done and step_count < 500:  # Safety limit
        # Agent decides action
        action = agent.decide(obs)
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        
        # Record frame and metrics
        frames.append(env.render_grid())
        metrics["rewards"].append(reward)
        metrics["battery"].append(obs["battery"])
        metrics["dist_pickup"].append(manhattan_distance(obs["robot_pos"], obs["pickup_pos"]))
        metrics["dist_dropoff"].append(manhattan_distance(obs["robot_pos"], obs["dropoff_pos"]))
        
        # Print progress
        if step_count <= 5 or step_count % 10 == 0 or terminated or truncated:
            status = "✓ SUCCESS" if terminated else "CONTINUE" if not truncated else "✗ TIMEOUT"
            print(f"Step {step_count:3d}: Action={action:5s} | Reward={reward:+6.2f} | "
                  f"Total={total_reward:+7.2f} | Pos={obs['robot_pos']} | "
                  f"Item={obs['has_item']} | Battery={obs['battery']:3d}")
        
        done = terminated or truncated
    
    # Print episode summary
    print()
    print("=" * 60)
    print("EPISODE SUMMARY")
    print("=" * 60)
    print(f"Total Reward:    {total_reward:+.2f}")
    print(f"Final Battery:   {obs['battery']}")
    print(f"Episode Length:  {step_count} steps")
    print(f"Success:         {'YES ✓' if obs['has_item'] == False and obs['robot_pos'] == obs['dropoff_pos'] else 'NO ✗'}")
    print("=" * 60)
    print()
    
    # Replay animation with metrics
    print("Launching animation replay...")
    print("Controls: SPACE to pause/resume, LEFT/RIGHT arrows to step through")
    replay_animation(frames, metrics=metrics, interval_ms=200)
    
    return {
        "total_reward": total_reward,
        "final_battery": obs["battery"],
        "episode_length": step_count,
        "success": obs["has_item"] == False and obs["robot_pos"] == obs["dropoff_pos"],
    }


if __name__ == "__main__":
    results = run_episode(randomize=True)
    print("\nFinal Results:")
    print(f"  Total Reward: {results['total_reward']:.2f}")
    print(f"  Final Battery: {results['final_battery']}")
    print(f"  Episode Length: {results['episode_length']}")
    print(f"  Success: {results['success']}")
