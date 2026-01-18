"""
algo/rl.py

Improved Actor-Critic with return normalization and proper gradient flow.
"""

import numpy as np
from typing import Dict, Optional
import time


class ActorCritic:
    """
    Actor-Critic with critical fixes for maze navigation:
    - Return normalization for stable value learning
    - Reward clipping to prevent outliers
    - Larger actor learning rate
    - Minimal exploration annealing
    """
    
    def __init__(
        self,
        policy,
        evaluator,
        env,
        agent,
        actor_lr: float = 0.01,
        critic_lr: float = 0.01,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        episodes_per_iteration: int = 10,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        seed: Optional[int] = None
    ):
        self.policy = policy
        self.evaluator = evaluator
        self.env = env
        self.agent = agent
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.episodes_per_iteration = episodes_per_iteration
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        self.rng = np.random.RandomState(seed)
        
        self.iteration = 0
        self.total_episodes = 0
        
        self.history = {
            'iteration': [],
            'mean_reward': [],
            'std_reward': [],
            'success_rate': [],
            'mean_steps': [],
            'actor_loss': [],
            'critic_loss': [],
            'entropy': [],
            'mean_distance_to_goal': []
        }
    
    def _collect_episode(self) -> Dict:
        self.env.reset()
        start_pos = self.env.get_start()
        self.agent.reset(start_pos)
        goal_x, goal_y = self.env.get_goal()
        
        observations = []
        actions = []
        rewards = []
        values = []
        distances = []
        
        prev_distance = np.sqrt((self.agent.x - goal_x)**2 + (self.agent.y - goal_y)**2)
        
        for step in range(self.evaluator.max_steps):
            obs = self.agent.get_observation(
                distance_fn=self.env.ray_distance,
                include_velocity=self.evaluator.include_velocity_obs
            )
            
            action = self.policy.forward(obs, deterministic=False)
            value = self.policy.get_value(obs)
            
            observations.append(obs.copy())
            actions.append(action.copy())
            values.append(value)
            
            self.agent.apply_action(action)
            self.agent.update(collision_fn=self.env.collides)
            
            current_distance = np.sqrt((self.agent.x - goal_x)**2 + (self.agent.y - goal_y)**2)
            distances.append(current_distance)
            
            reward = self._compute_reward(current_distance, prev_distance,
                                         current_distance < self.evaluator.goal_threshold)
            
            # Clip reward to prevent outliers
            reward = np.clip(reward, -50, 150)
            
            rewards.append(reward)
            prev_distance = current_distance
            
            if current_distance < self.evaluator.goal_threshold:
                break
        
        return {
            'observations': np.array(observations),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'values': np.array(values),
            'distances': np.array(distances),
            'success': current_distance < self.evaluator.goal_threshold,
            'steps': len(rewards),
            'final_distance': current_distance
        }
    
    def _compute_reward(self, current_distance, prev_distance, reached_goal):
        reward = 0.0
        
        # Distance shaping
        delta = prev_distance - current_distance
        reward += self.evaluator.distance_reward_scale * delta
        
        # Step penalty
        reward += self.evaluator.step_penalty
        
        # Goal bonus
        if reached_goal:
            reward += self.evaluator.goal_reward
        
        return reward
    
    def _compute_gae(self, rewards, values, gamma, gae_lambda):
        T = len(rewards)
        advantages = np.zeros(T)
        returns = np.zeros(T)
        
        gae = 0.0
        next_value = 0.0
        
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]
        
        return advantages, returns
    
    def step(self) -> Dict:
        trajectories = []
        episode_rewards = []
        episode_successes = []
        episode_steps = []
        episode_distances = []
        
        for _ in range(self.episodes_per_iteration):
            traj = self._collect_episode()
            trajectories.append(traj)
            episode_rewards.append(np.sum(traj['rewards']))
            episode_successes.append(traj['success'])
            episode_steps.append(traj['steps'])
            episode_distances.append(traj['final_distance'])
        
        # Compute advantages
        all_obs = []
        all_actions = []
        all_advantages = []
        all_returns = []
        all_values = []
        
        for traj in trajectories:
            advantages, returns = self._compute_gae(
                traj['rewards'], traj['values'], self.gamma, self.gae_lambda
            )
            all_obs.append(traj['observations'])
            all_actions.append(traj['actions'])
            all_advantages.append(advantages)
            all_returns.append(returns)
            all_values.append(traj['values'])
        
        batch_obs = np.concatenate(all_obs, axis=0)
        batch_actions = np.concatenate(all_actions, axis=0)
        batch_advantages = np.concatenate(all_advantages, axis=0)
        batch_returns = np.concatenate(all_returns, axis=0)
        batch_values = np.concatenate(all_values, axis=0)
        
        # Update return normalization
        self.policy.ret_rms.update(batch_returns.reshape(-1, 1))
        
        # Normalize advantages
        batch_advantages = (batch_advantages - np.mean(batch_advantages)) / (np.std(batch_advantages) + 1e-8)
        batch_advantages = np.clip(batch_advantages, -10, 10)
        
        # Normalize returns for critic
        batch_returns_norm = self.policy.ret_rms.normalize(batch_returns.reshape(-1, 1)).flatten()
        
        # Evaluate policy
        log_probs, values_norm, entropy = self.policy.evaluate_actions(batch_obs, batch_actions)
        
        # Losses
        actor_loss = -np.mean(log_probs * batch_advantages)
        critic_loss = np.mean((values_norm - batch_returns_norm) ** 2)
        entropy_loss = -np.mean(entropy)
        
        # Update
        self._update_parameters(batch_obs, batch_actions, batch_advantages, batch_returns_norm)
        
        self.iteration += 1
        self.total_episodes += self.episodes_per_iteration
        
        summary = {
            'iteration': self.iteration,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': np.mean(episode_successes),
            'mean_steps': np.mean(episode_steps),
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy': -entropy_loss,
            'mean_distance_to_goal': np.mean(episode_distances)
        }
        
        for key in self.history.keys():
            if key in summary:
                self.history[key].append(summary[key])
        
        return summary
    
    def _update_parameters(self, obs, actions, advantages, returns_norm):
        N = len(obs)
        
        # Actor gradient
        grad_W_actor = np.zeros_like(self.policy.W_actor)
        grad_b_actor = np.zeros_like(self.policy.b_actor)
        grad_log_std = np.zeros_like(self.policy.log_std)
        
        for i in range(N):
            obs_norm = self.policy._normalize_obs(obs[i])
            mean = np.tanh(self.policy.W_actor @ obs_norm + self.policy.b_actor)
            std = np.exp(self.policy.log_std)
            
            action_diff = actions[i] - mean
            tanh_grad = 1.0 - mean ** 2
            delta = (action_diff / (std ** 2)) * tanh_grad
            
            grad_W_actor += -advantages[i] * np.outer(delta, obs_norm)
            grad_b_actor += -advantages[i] * delta
            grad_log_std += -advantages[i] * ((action_diff ** 2) / (std ** 2) - 1)
        
        grad_W_actor /= N
        grad_b_actor /= N
        grad_log_std /= N
        
        # Clip
        grad_norm = np.sqrt(np.sum(grad_W_actor**2) + np.sum(grad_b_actor**2) + np.sum(grad_log_std**2))
        if grad_norm > self.max_grad_norm:
            grad_W_actor *= self.max_grad_norm / grad_norm
            grad_b_actor *= self.max_grad_norm / grad_norm
            grad_log_std *= self.max_grad_norm / grad_norm
        
        # Update actor
        self.policy.W_actor -= self.actor_lr * grad_W_actor
        self.policy.b_actor -= self.actor_lr * grad_b_actor
        self.policy.log_std -= self.actor_lr * 0.1 * grad_log_std  # Slower log_std updates
        
        # Critic gradient
        grad_W_critic = np.zeros_like(self.policy.W_critic)
        grad_b_critic = 0.0
        
        for i in range(N):
            obs_norm = self.policy._normalize_obs(obs[i])
            value = self.policy.W_critic @ obs_norm + self.policy.b_critic
            value_error = value - returns_norm[i]
            
            grad_W_critic += value_error * obs_norm
            grad_b_critic += value_error
        
        grad_W_critic /= N
        grad_b_critic /= N
        
        # Update critic
        self.policy.W_critic -= self.critic_lr * grad_W_critic
        self.policy.b_critic -= self.critic_lr * grad_b_critic
    
    def train(self, num_iterations: int, verbose: bool = True) -> Dict:
        if verbose:
            print(f"Starting Actor-Critic training for {num_iterations} iterations")
            print("=" * 70)
        
        start_time = time.time()
        
        for it in range(num_iterations):
            summary = self.step()
            
            if verbose:
                print(
                    f"Iter {summary['iteration']:3d} | "
                    f"Reward: {summary['mean_reward']:7.2f} ± {summary['std_reward']:5.2f} | "
                    f"Success: {summary['success_rate']:5.1%} | "
                    f"Dist: {summary['mean_distance_to_goal']:5.2f} | "
                    f"A_loss: {summary['actor_loss']:.3f} | "
                    f"C_loss: {summary['critic_loss']:.3f}"
                )
        
        total_time = time.time() - start_time
        
        if verbose:
            print("=" * 70)
            print(f"Training complete in {total_time:.1f}s")
            print(f"Final success rate: {self.history['success_rate'][-1]:.1%}")
        
        return {
            'policy': self.policy,
            'history': self.history,
            'total_time': total_time,
            'total_episodes': self.total_episodes
        }
    
    def get_policy(self):
        return self.policy
    
    def get_history(self):
        return self.history
