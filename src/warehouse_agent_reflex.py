import random
from typing import Dict, List, Tuple, Union
from warehouse_env import WarehouseEnv

Action = Union[int, str]


class ReflexAgent:
    """
    A simple reflex agent for the warehouse environment.
    Uses rule-based decision making based on current state and observations.
    """

    def __init__(self, env: WarehouseEnv = None):
        self.env = env
        self.actions = WarehouseEnv.ACTIONS
        self.move_actions = ["N", "E", "S", "W"]

    def decide(self, observation: Dict[str, object]) -> Action:
        """
        Decide on the next action based on the current observation.

        Rules (in priority order):
        1. If at pickup location and no item: pick
        2. If at dropoff location and carrying item: drop
        3. If carrying item: move toward dropoff (avoiding walls)
        4. If not carrying item: move toward pickup (avoiding walls)
        5. Otherwise: choose a random valid action
        """
        robot_pos = observation["robot_pos"]
        has_item = observation["has_item"]
        pickup_pos = observation["pickup_pos"]
        dropoff_pos = observation["dropoff_pos"]
        local_grid = observation["local_grid"]

        # Rule 1: At pickup location and no item -> pick
        if robot_pos == pickup_pos and not has_item:
            return "PICK"

        # Rule 2: At dropoff location and carrying item -> drop
        if robot_pos == dropoff_pos and has_item:
            return "DROP"

        # Determine target position based on carry state
        if has_item:
            target_pos = dropoff_pos
        else:
            target_pos = pickup_pos

        # Rule 3 & 4: Move toward target, avoiding walls
        if target_pos is not None:
            direction = self._get_direction_to_target(robot_pos, target_pos, local_grid)
            if direction is not None:
                return direction

        # Rule 5: No applicable rule -> try any valid (non-wall) move
        valid_moves = self._get_valid_moves(local_grid)
        if valid_moves:
            return random.choice(valid_moves)
        
        # If all moves are blocked, wait
        return "WAIT"

    def _get_direction_to_target(
        self, current_pos: Tuple[int, int], target_pos: Tuple[int, int], local_grid: List[str]
    ) -> Union[str, None]:
        """
        Determine the direction to move toward the target, avoiding walls.
        Uses a priority system:
        1. Move toward target if not blocked
        2. If blocked, try perpendicular directions (weighted by target direction)
        3. If all perpendicular blocked, try any available move
        """
        cr, cc = current_pos
        tr, tc = target_pos

        dr = tr - cr
        dc = tc - cc

        # If we're at the target, no need to move
        if dr == 0 and dc == 0:
            return None

        # Local grid is centered on robot
        view_radius = len(local_grid) // 2
        center = view_radius
        
        # Mapping of directions to local_grid offsets
        direction_map = {
            "N": (center - 1, center),
            "S": (center + 1, center),
            "W": (center, center - 1),
            "E": (center, center + 1),
        }
        
        # Check if a direction is blocked
        def is_blocked(direction: str) -> bool:
            r, c = direction_map[direction]
            return r < 0 or r >= len(local_grid) or c < 0 or c >= len(local_grid[0]) or local_grid[r][c] == "#"
        
        # Primary directions (toward target)
        primary = []
        if dr < 0 and not is_blocked("N"):
            primary.append(("N", 2))  # Weight 2 for moving toward target
        if dr > 0 and not is_blocked("S"):
            primary.append(("S", 2))
        if dc < 0 and not is_blocked("W"):
            primary.append(("W", 2))
        if dc > 0 and not is_blocked("E"):
            primary.append(("E", 2))
        
        # If primary direction available, return it
        if primary:
            return max(primary, key=lambda x: x[1])[0]
        
        # Secondary directions (perpendicular to target)
        secondary = []
        if dc < 0 and not is_blocked("W"):
            secondary.append(("W", 1))
        if dc > 0 and not is_blocked("E"):
            secondary.append(("E", 1))
        if dr < 0 and not is_blocked("N"):
            secondary.append(("N", 1))
        if dr > 0 and not is_blocked("S"):
            secondary.append(("S", 1))
        
        # If secondary direction available, return randomly from all secondary
        if secondary:
            return random.choice(secondary)[0]
        
        # Last resort: try any valid move
        valid_moves = [d for d in ["N", "S", "E", "W"] if not is_blocked(d)]
        if valid_moves:
            return random.choice(valid_moves)

        return None

    def _get_valid_moves(self, local_grid: List[str]) -> List[str]:
        """
        Get all valid movement directions that don't lead to walls.
        """
        view_radius = len(local_grid) // 2
        center = view_radius
        valid = []
        
        if local_grid[center - 1][center] != "#":
            valid.append("N")
        if local_grid[center + 1][center] != "#":
            valid.append("S")
        if local_grid[center][center - 1] != "#":
            valid.append("W")
        if local_grid[center][center + 1] != "#":
            valid.append("E")
        
        # Also consider non-movement actions
        valid.extend(["WAIT", "PICK", "DROP"])
        
        return valid

    def _random_action(self) -> Action:
        """Return a random movement action from valid moves, or random from all actions."""
        return random.choice(self.actions)


def test_agent():
    """Test the reflex agent with the warehouse environment."""
    env = WarehouseEnv()
    agent = ReflexAgent(env)

    obs = env.reset()
    print("Starting test of ReflexAgent...")
    print(env.render_with_legend())
    print()

    total_reward = 0
    for step in range(200):  # Run for up to 200 steps
        action = agent.decide(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step < 20 or step % 10 == 0 or terminated or truncated:
            print(f"Step {step + 1}: Action={action}, Reward={reward:.2f}, Total={total_reward:.2f}")
            print(f"  Position: {obs['robot_pos']}, Has Item: {obs['has_item']}, Battery: {obs['battery']}")

        if terminated:
            print(f"\n✓ Goal achieved! Task completed in {obs['steps']} steps with reward {total_reward:.2f}")
            break

        if truncated:
            print(f"\n✗ Episode truncated. Steps: {obs['steps']}, Reward: {total_reward:.2f}")
            break

    print("\nFinal state:")
    print(env.render_with_legend())


if __name__ == "__main__":
    test_agent()
