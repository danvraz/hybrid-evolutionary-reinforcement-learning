"""
policy/actor_critic_policy.py

Improved Actor-Critic with proper action scaling and observation normalization.
"""

import numpy as np
from typing import Tuple, Optional


class RunningMeanStd:
    """Tracks running mean and std for normalization."""
    
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = M2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class ActorCriticPolicy:
    """
    Actor-Critic with proper action scaling and normalization.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        init_scale: float = 0.1,
        log_std_init: float = 0.0,
        seed: Optional[int] = None
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        rng = np.random.RandomState(seed)
        
        # Actor
        self.W_actor = rng.randn(action_dim, obs_dim) * init_scale
        self.b_actor = np.zeros(action_dim)
        self.log_std = np.ones(action_dim) * log_std_init
        
        # Critic
        self.W_critic = rng.randn(obs_dim) * init_scale
        self.b_critic = 0.0
        
        # Normalization
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.ret_rms = RunningMeanStd(shape=(1,))
        self.normalize_obs = True
        self.normalize_ret = True
    
    def _normalize_obs(self, obs):
        if self.normalize_obs:
            return self.obs_rms.normalize(obs)
        return obs
    
    def forward(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Sample action - note: deterministic is ignored for now to maintain exploration."""
        obs_norm = self._normalize_obs(obs)
        mean = np.tanh(self.W_actor @ obs_norm + self.b_actor)
        
        # Always use stochastic policy (exploration is critical)
        std = np.exp(self.log_std)
        noise = np.random.randn(self.action_dim) * std
        action = np.clip(mean + noise, -1.0, 1.0)
        
        return action
    
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        # For evaluation, still use some exploration
        return self.forward(obs, deterministic=False)
    
    def get_value(self, obs: np.ndarray) -> float:
        obs_norm = self._normalize_obs(obs)
        value = self.W_critic @ obs_norm + self.b_critic
        
        # Denormalize value if using return normalization
        if self.normalize_ret:
            value = value * np.sqrt(self.ret_rms.var[0] + 1e-8) + self.ret_rms.mean[0]
        
        return value
    
    def evaluate_actions(
        self,
        obs: np.ndarray,
        actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = len(obs)
        
        # Update obs stats
        self.obs_rms.update(obs)
        
        log_probs = np.zeros(N)
        values = np.zeros(N)
        entropy_vals = np.zeros(N)
        
        for i in range(N):
            obs_norm = self._normalize_obs(obs[i])
            
            # Actor
            mean = np.tanh(self.W_actor @ obs_norm + self.b_actor)
            std = np.exp(self.log_std)
            
            # Critic
            value = self.W_critic @ obs_norm + self.b_critic
            values[i] = value
            
            # Log prob
            action_diff = actions[i] - mean
            log_prob = -0.5 * np.sum(
                (action_diff / std) ** 2 + 
                2 * self.log_std + 
                np.log(2 * np.pi)
            )
            log_probs[i] = log_prob
            
            # Entropy
            entropy_vals[i] = 0.5 * np.sum(self.log_std + np.log(2 * np.pi * np.e) + 0.5)
        
        return log_probs, values, entropy_vals
