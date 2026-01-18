"""
eval/rollout.py

Pure episode evaluation for maze navigation task.

This module executes a single episode rollout given an environment,
agent, and policy. It computes task-specific rewards but does NOT
perform learning, mutation, or policy updates.

Responsibilities:
- Episode execution (reset → step → termination)
- Reward computation (distance shaping + goal bonus)
- Metric collection (success, steps, cumulative reward)

This is used by:
- Evolutionary fitness evaluation
- RL training loops
- Offline policy assessment
"""

import numpy as np
from typing import Dict, Callable, Optional


class RolloutEvaluator:
    """
    Episode evaluator for continuous maze navigation.
    
    Executes a single episode and computes reward based on:
    - Distance-to-goal shaping (dense feedback)
    - Step penalty (encourages efficiency)
    - Terminal goal bonus (sparse reward)
    
    This class does NOT:
    - Modify the policy
    - Perform learning updates
    - Store episode history (returns summary only)
    """
    
    def __init__(
        self,
        max_steps: int = 500,
        goal_threshold: float = 0.5,
        step_penalty: float = -0.01,
        goal_reward: float = 100.0,
        distance_reward_scale: float = 1.0,
        include_velocity_obs: bool = True
    ):
        """
        Initialize evaluator with reward hyperparameters.
        
        Args:
            max_steps: Maximum episode length
            goal_threshold: Distance to goal for success (continuous units)
            step_penalty: Penalty per timestep (encourages efficiency)
            goal_reward: Bonus for reaching goal
            distance_reward_scale: Scale factor for distance shaping
            include_velocity_obs: Whether to include velocity in observations
        """
        self.max_steps = max_steps
        self.goal_threshold = goal_threshold
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward
        self.distance_reward_scale = distance_reward_scale
        self.include_velocity_obs = include_velocity_obs
    
    def evaluate(
        self,
        env,  # MazeEnv instance
        agent,  # Agent instance
        policy: Callable[[np.ndarray], np.ndarray],
        seed: Optional[int] = None
    ) -> Dict:
        """
        Execute single episode and return metrics.
        
        Episode flow:
        1. Reset environment and agent
        2. Loop until termination:
           - Get observation from agent sensors
           - Query policy for action
           - Apply action to agent
           - Update agent physics
           - Compute reward
        3. Return summary metrics
        
        Args:
            env: Maze environment (already constructed)
            agent: Embodied agent (already constructed)
            policy: Function obs -> action (callable)
            seed: Optional seed for episode determinism
        
        Returns:
            metrics: Dictionary containing:
                - success (bool): Whether goal was reached
                - steps (int): Number of steps taken
                - cumulative_reward (float): Total reward
                - final_distance_to_goal (float): Distance at termination
                - mean_distance_to_goal (float): Average distance over episode
                - goal_reached_step (int or None): Step when goal reached
        """
        # Set seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset environment (generates new maze, start, goal)
        env.reset()
        
        # Reset agent to start position
        start_pos = env.get_start()
        agent.reset(start_pos)
        
        # Get goal position (fixed for this episode)
        goal_x, goal_y = env.get_goal()
        
        # Episode state
        cumulative_reward = 0.0
        total_distance = 0.0  # For mean computation
        prev_distance = self._distance_to_goal(agent, goal_x, goal_y)
        
        success = False
        goal_reached_step = None
        
        # Execute episode
        for step in range(self.max_steps):
            # Get observation from agent sensors
            obs = agent.get_observation(
                distance_fn=env.ray_distance,
                include_velocity=self.include_velocity_obs
            )
            
            # Query policy (external, no learning here)
            action = policy(obs)
            
            # Validate action shape
            assert len(action) == agent.action_dim, \
                f"Policy returned action of shape {action.shape}, expected ({agent.action_dim},)"
            
            # Apply action to agent
            agent.apply_action(action)
            
            # Update agent physics (with collision resolution)
            agent.update(collision_fn=env.collides)
            
            # Compute current distance to goal
            current_distance = self._distance_to_goal(agent, goal_x, goal_y)
            total_distance += current_distance
            
            # Compute reward
            reward = self._compute_reward(
                current_distance=current_distance,
                prev_distance=prev_distance,
                reached_goal=(current_distance < self.goal_threshold)
            )
            
            cumulative_reward += reward
            prev_distance = current_distance
            
            # Check termination
            if current_distance < self.goal_threshold:
                success = True
                goal_reached_step = step + 1
                break
        
        # Final distance
        final_distance = self._distance_to_goal(agent, goal_x, goal_y)
        
        # Compute mean distance
        steps_taken = step + 1
        mean_distance = total_distance / steps_taken
        
        # Return metrics
        return {
            'success': success,
            'steps': steps_taken,
            'cumulative_reward': cumulative_reward,
            'final_distance_to_goal': final_distance,
            'mean_distance_to_goal': mean_distance,
            'goal_reached_step': goal_reached_step
        }
    
    def _distance_to_goal(self, agent, goal_x: float, goal_y: float) -> float:
        """
        Compute Euclidean distance from agent to goal.
        
        Args:
            agent: Agent instance with .x and .y attributes
            goal_x: Goal X coordinate
            goal_y: Goal Y coordinate
        
        Returns:
            distance: Euclidean distance
        """
        return np.sqrt((agent.x - goal_x)**2 + (agent.y - goal_y)**2)
    
    def _compute_reward(
        self,
        current_distance: float,
        prev_distance: float,
        reached_goal: bool
    ) -> float:
        """
        Compute reward for single timestep.
        
        Reward components:
        1. Distance shaping: reward for getting closer to goal
        2. Step penalty: small penalty to encourage efficiency
        3. Goal bonus: large reward for reaching goal
        
        Args:
            current_distance: Distance to goal after action
            prev_distance: Distance to goal before action
            reached_goal: Whether goal threshold was reached
        
        Returns:
            reward: Scalar reward value
        """
        reward = 0.0
        
        # Distance shaping (potential-based, dense feedback)
        # Reward for getting closer, penalty for getting farther
        distance_delta = prev_distance - current_distance
        reward += self.distance_reward_scale * distance_delta
        
        # Step penalty (encourages shorter paths)
        reward += self.step_penalty
        
        # Goal bonus (sparse, large reward)
        if reached_goal:
            reward += self.goal_reward
        
        return reward
    
    def evaluate_batch(
        self,
        env,
        agent,
        policies: list,
        seeds: Optional[list] = None
    ) -> list:
        """
        Evaluate multiple policies in sequence.
        
        Useful for evolutionary fitness evaluation where we need
        to assess many parameter vectors.
        
        Args:
            env: Maze environment
            agent: Embodied agent
            policies: List of policy callables
            seeds: Optional list of seeds (one per policy)
        
        Returns:
            results: List of metric dictionaries
        """
        if seeds is None:
            seeds = [None] * len(policies)
        
        assert len(seeds) == len(policies), \
            "Number of seeds must match number of policies"
        
        results = []
        for policy, seed in zip(policies, seeds):
            metrics = self.evaluate(env, agent, policy, seed=seed)
            results.append(metrics)
        
        return results


# Validation
if __name__ == "__main__":
    # Mock imports (would normally come from other modules)
    class MockEnv:
        def __init__(self):
            self.start = (1.0, 1.0)
            self.goal = (5.0, 5.0)
        
        def reset(self):
            pass
        
        def get_start(self):
            return self.start
        
        def get_goal(self):
            return self.goal
        
        def collides(self, x, y, r):
            return False
        
        def ray_distance(self, x, y, dx, dy):
            return 5.0
    
    class MockAgent:
        def __init__(self):
            self.x = 1.0
            self.y = 1.0
            self.action_dim = 2
        
        def reset(self, pos):
            self.x, self.y = pos
        
        def get_observation(self, distance_fn, include_velocity):
            return np.zeros(10)  # Mock observation
        
        def apply_action(self, action):
            # Simple physics: move in action direction
            self.x += action[0] * 0.1
            self.y += action[1] * 0.1
        
        def update(self, collision_fn):
            pass
    
    # Mock policy (moves toward goal)
    def greedy_policy(obs):
        # Always move toward (5, 5)
        return np.array([0.5, 0.5])
    
    print("RolloutEvaluator - Pure episode evaluation")
    print("=" * 60)
    
    # Create evaluator
    evaluator = RolloutEvaluator(
        max_steps=100,
        goal_threshold=0.5,
        step_penalty=-0.01,
        goal_reward=100.0
    )
    
    # Create mock environment and agent
    env = MockEnv()
    agent = MockAgent()
    
    # Run evaluation
    metrics = evaluator.evaluate(env, agent, greedy_policy, seed=42)
    
    print("Episode metrics:")
    print(f"  Success: {metrics['success']}")
    print(f"  Steps: {metrics['steps']}")
    print(f"  Cumulative reward: {metrics['cumulative_reward']:.2f}")
    print(f"  Final distance: {metrics['final_distance_to_goal']:.2f}")
    print(f"  Mean distance: {metrics['mean_distance_to_goal']:.2f}")
    print(f"  Goal reached at step: {metrics['goal_reached_step']}")
    
    print()
    print("✓ Evaluator implementation complete")
    print("✓ Ready for integration with evolution and RL")
    print("✓ No learning, no mutation, pure evaluation")
