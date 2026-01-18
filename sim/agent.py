# ============================================================
# LOCKED MODULE
# Embodied agent with continuous physics + ray sensing
# DO NOT MODIFY unless fixing a confirmed bug.
# This file defines the agent body for all experiments.
# ============================================================

"""
Embodied 2D agent with continuous physics and ray-based sensing.

This module implements the agent's physical body and sensory apparatus.
It contains NO policy logic, NO learning, and NO reward computation.

Key responsibilities:
- Position and velocity dynamics
- Wall collision resolution via environment queries
- Ray-based distance sensing (partial observability)
- Deterministic state updates

The agent is a "sensor-equipped robot body" — nothing more.
"""

import numpy as np
from typing import Tuple, Callable


class Agent:
    """
    Continuous 2D agent with physics-based movement and ray sensors.

    The agent maintains its own state (position, velocity) and provides
    methods for:
    - Applying control actions
    - Updating physics (with collision resolution)
    - Sensing the environment via distance rays

    The agent does NOT:
    - Choose its own actions (policy is external)
    - Compute rewards (environment's responsibility)
    - Store episode history (evaluator's responsibility)
    """

    def __init__(
        self,
        position: Tuple[float, float],
        radius: float = 0.2,
        max_speed: float = 2.0,
        num_rays: int = 8,
        ray_max_distance: float = 5.0,
        dt: float = 0.05
    ):
        """
        Initialize agent state and sensor configuration.

        Args:
            position: Initial (x, y) position
            radius: Agent collision radius
            max_speed: Maximum velocity magnitude
            num_rays: Number of distance sensors
            ray_max_distance: Maximum sensing range
            dt: Physics timestep
        """
        # State
        self.x, self.y = position
        self.vx, self.vy = 0.0, 0.0

        # Physical properties
        self.radius = radius
        self.max_speed = max_speed
        self.dt = dt

        # Sensor configuration
        self.num_rays = num_rays
        self.ray_max_distance = ray_max_distance

        # Precompute ray angles (even distribution)
        self.ray_angles = np.linspace(
            0.0, 2.0 * np.pi, num_rays, endpoint=False
        )

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def reset(self, position: Tuple[float, float]):
        """Reset agent state."""
        self.x, self.y = position
        self.vx, self.vy = 0.0, 0.0

    # --------------------------------------------------
    # Control
    # --------------------------------------------------

    def apply_action(self, action: np.ndarray):
        """
        Apply control action to velocity.

        Action is interpreted as a velocity delta.
        """
        assert action.shape == (2,), "Action must be 2D"

        self.vx += action[0]
        self.vy += action[1]

        speed = np.hypot(self.vx, self.vy)
        if speed > self.max_speed:
            self.vx = (self.vx / speed) * self.max_speed
            self.vy = (self.vy / speed) * self.max_speed

    # --------------------------------------------------
    # Physics update
    # --------------------------------------------------

    def update(self, collision_fn: Callable[[float, float, float], bool]):
        """
        Update agent position with collision handling.

        Args:
            collision_fn: function(x, y, radius) -> bool
        """
        new_x = self.x + self.vx * self.dt
        new_y = self.y + self.vy * self.dt

        if collision_fn(new_x, new_y, self.radius):
            x_ok = not collision_fn(new_x, self.y, self.radius)
            y_ok = not collision_fn(self.x, new_y, self.radius)

            if x_ok and not y_ok:
                self.x = new_x
                self.vy = 0.0
            elif y_ok and not x_ok:
                self.y = new_y
                self.vx = 0.0
            elif x_ok and y_ok:
                self.x = new_x
                self.y = new_y
            else:
                self.vx = 0.0
                self.vy = 0.0
        else:
            self.x = new_x
            self.y = new_y

    # --------------------------------------------------
    # Sensing
    # --------------------------------------------------

    def sense(self, distance_fn: Callable[[float, float, float, float], float]) -> np.ndarray:
        """
        Ray-based distance sensing.

        Args:
            distance_fn: function(x, y, dx, dy) -> distance

        Returns:
            Normalized distances in [0, 1], closer = higher
        """
        obs = np.zeros(self.num_rays, dtype=np.float32)

        for i, angle in enumerate(self.ray_angles):
            dx = np.cos(angle)
            dy = np.sin(angle)

            dist = distance_fn(self.x, self.y, dx, dy)
            dist = min(dist, self.ray_max_distance)

            obs[i] = 1.0 - (dist / self.ray_max_distance)

        return obs

    def get_observation(
        self,
        distance_fn: Callable[[float, float, float, float], float],
        include_velocity: bool = True
    ) -> np.ndarray:
        """
        Full observation vector.

        Observation = [ray_distances] + optional [vx, vy]
        """
        rays = self.sense(distance_fn)

        if include_velocity:
            vel = np.array([
                self.vx / self.max_speed,
                self.vy / self.max_speed
            ], dtype=np.float32)
            return np.concatenate([rays, vel])

        return rays

    # --------------------------------------------------
    # Introspection
    # --------------------------------------------------

    @property
    def observation_dim(self) -> int:
        return self.num_rays + 2

    @property
    def observation_dim_no_vel(self) -> int:
        return self.num_rays

    @property
    def action_dim(self) -> int:
        return 2

    def get_state(self) -> dict:
        return {
            "position": (self.x, self.y),
            "velocity": (self.vx, self.vy),
            "speed": np.hypot(self.vx, self.vy),
            "radius": self.radius,
            "num_rays": self.num_rays
        }


# --------------------------------------------------
# Self-test (safe to run)
# --------------------------------------------------

if __name__ == "__main__":
    print("✓ Embodied agent self-test")

    agent = Agent(position=(1.0, 1.0))

    def no_collision(x, y, r):
        return False

    def infinite_distance(x, y, dx, dy):
        return 100.0

    obs = agent.get_observation(infinite_distance)
    assert obs.shape[0] == agent.observation_dim

    agent.apply_action(np.array([0.5, 0.2]))
    agent.update(no_collision)

    print("✓ Agent OK")
