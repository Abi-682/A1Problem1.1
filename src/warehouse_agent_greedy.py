import random
from collections import deque
from typing import Dict, List, Tuple, Union

from warehouse_env import WarehouseEnv

Action = Union[int, str]


class GreedyManhattanAgent:
    """
    Greedy agent that moves to minimize Manhattan distance to current goal.

    Behavior:
    - If not carrying an item: target = pickup tile
    - If carrying an item: target = dropoff tile
    - If on target and appropriate state -> PICK or DROP
    - Choose move (N/E/S/W) that reduces Manhattan distance
    - If no move reduces distance, choose a random valid move
    - Loop detector: remember last N=10 positions; if revisited, trigger
      a short random escape for a few steps.
    """

    def __init__(self, env: WarehouseEnv, history_len: int = 10, escape_steps: int = 3):
        self.env = env
        self.actions = env.ACTIONS
        self.move_actions = ["N", "E", "S", "W"]
        self.history = deque(maxlen=history_len)
        self.escape_duration = escape_steps
        self.escape_steps_remaining = 0

    def decide(self, observation: Dict[str, object]) -> Action:
        """
        Return an action string or int compatible with the environment.
        Expects the observation format produced by `WarehouseEnv._observe()`.
        """
        robot_pos: Tuple[int, int] = observation["robot_pos"]
        has_item: bool = observation["has_item"]
        pickup_pos = observation.get("pickup_pos")
        dropoff_pos = observation.get("dropoff_pos")
        local_grid: List[str] | None = observation.get("local_grid")

        # Loop detection: if we've been here recently, enter escape mode
        if robot_pos in self.history:
            self.escape_steps_remaining = self.escape_duration
        # Append current position after checking (so immediate revisit counts)
        self.history.append(robot_pos)

        # If in escape mode, take random valid moves for a few steps
        if self.escape_steps_remaining > 0:
            self.escape_steps_remaining -= 1
            mv = self._random_valid_move(local_grid)
            if mv is not None:
                return mv

        # If at pickup and not carrying -> PICK
        if pickup_pos is not None and robot_pos == pickup_pos and not has_item:
            return "PICK"

        # If at dropoff and carrying -> DROP
        if dropoff_pos is not None and robot_pos == dropoff_pos and has_item:
            return "DROP"

        # Determine goal
        goal = dropoff_pos if has_item else pickup_pos
        if goal is None:
            # No known goal: fall back to random valid move
            mv = self._random_valid_move(local_grid)
            return mv if mv is not None else random.choice(self.actions)

        # Compute current Manhattan distance
        cur_dist = abs(robot_pos[0] - goal[0]) + abs(robot_pos[1] - goal[1])

        # Find moves that reduce Manhattan distance
        best_moves: List[str] = []
        best_dist = cur_dist
        for m in self.move_actions:
            dr, dc = self.env.MOVE_DELTAS[m]
            nr, nc = robot_pos[0] + dr, robot_pos[1] + dc
            # Skip moves into walls
            if self.env._is_wall(nr, nc):
                continue
            d = abs(nr - goal[0]) + abs(nc - goal[1])
            if d < best_dist:
                best_dist = d
                best_moves = [m]
            elif d == best_dist and d < cur_dist:
                best_moves.append(m)

        if best_moves:
            # If several equivalent moves, pick one at random to break ties
            return random.choice(best_moves)

        # No move reduces distance -> try any valid move (avoid walls)
        mv = self._random_valid_move(local_grid)
        if mv is not None:
            return mv

        # As last resort, wait
        return "WAIT"

    def _random_valid_move(self, local_grid: List[str] | None) -> Union[str, None]:
        """
        Choose a random valid movement action using the local grid if available.
        Returns None if no valid move found.
        """
        valid = []
        if local_grid:
            view_radius = len(local_grid) // 2
            center = view_radius
            # check bounds implicitly; local grid contains '#' for out-of-bounds
            if local_grid[center - 1][center] != "#":
                valid.append("N")
            if local_grid[center + 1][center] != "#":
                valid.append("S")
            if local_grid[center][center - 1] != "#":
                valid.append("W")
            if local_grid[center][center + 1] != "#":
                valid.append("E")

        # If no local info or no moves from local grid, use env methods to check
        if not valid:
            for m in self.move_actions:
                dr, dc = self.env.MOVE_DELTAS[m]
                r, c = self.history[-1] if self.history else self.env.start_pos
                nr, nc = r + dr, c + dc
                if not self.env._is_wall(nr, nc):
                    valid.append(m)

        return random.choice(valid) if valid else None


def test_agent():
    env = WarehouseEnv()
    agent = GreedyManhattanAgent(env)

    obs = env.reset(randomize=True)
    print("Starting GreedyManhattanAgent demo...")
    print(env.render_with_legend())
    total_reward = 0.0
    for step in range(300):
        action = agent.decide(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step < 10 or step % 20 == 0 or terminated or truncated:
            print(f"Step {step+1}: Action={action}, Reward={reward:.2f}, Total={total_reward:.2f}")
            print(f"  Pos={obs['robot_pos']}, HasItem={obs['has_item']}, Battery={obs['battery']}")
        if terminated:
            print(f"\n✓ Success in {obs['steps']} steps. Total reward {total_reward:.2f}")
            break
        if truncated:
            print(f"\n✗ Truncated after {obs['steps']} steps. Total reward {total_reward:.2f}")
            break


if __name__ == "__main__":
    test_agent()
