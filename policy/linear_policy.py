"""
policy/linear_policy.py

Linear policy with tanh activation for continuous control.

This module defines a simple neural policy that maps observations
to actions via a linear transformation with tanh squashing.

Policy form:
    action = tanh(W @ obs + b)

where:
    W ∈ ℝ^(action_dim × obs_dim)
    b ∈ ℝ^(action_dim)
    θ = [W.flatten(), b]  (flat parameter vector)

The policy is:
- Deterministic (no sampling)
- Differentiable (for potential RL use)
- Parameterized by a single flat vector θ

This class does NOT:
- Perform learning
- Compute gradients
- Access the environment
- Define fitness or reward
"""

import numpy as np
from typing import Optional


class LinearPolicy:
    """
    Linear policy with tanh activation.
    
    Maps observations to actions via:
        action = tanh(W @ obs + b)
    
    Parameters are stored as a single flat vector for efficient
    evolutionary search and gradient-based optimization.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        init_scale: float = 0.1,
        seed: Optional[int] = None
    ):
        """
        Initialize policy with random parameters.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            init_scale: Scale for random initialization
            seed: Random seed for reproducibility
        """
        assert obs_dim > 0 and action_dim > 0, \
            "Dimensions must be positive"
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Initialize random number generator
        rng = np.random.RandomState(seed)
        
        # Initialize weight matrix: (action_dim, obs_dim)
        self.W = rng.randn(action_dim, obs_dim) * init_scale
        
        # Initialize bias vector: (action_dim,)
        self.b = np.zeros(action_dim)
        
        # Compute parameter count
        self.num_params = action_dim * obs_dim + action_dim
    
    def forward(self, obs: np.ndarray) -> np.ndarray:
        """
        Compute action from observation (deterministic).
        
        Args:
            obs: Observation vector (obs_dim,)
        
        Returns:
            action: Action vector (action_dim,), clipped to [-1, 1]
        """
        assert obs.shape == (self.obs_dim,), \
            f"Expected observation shape ({self.obs_dim},), got {obs.shape}"
        
        # Linear transformation
        logits = self.W @ obs + self.b
        
        # Tanh activation (squashes to [-1, 1])
        action = np.tanh(logits)
        
        # Ensure numerical stability (clip to exact bounds)
        action = np.clip(action, -1.0, 1.0)
        
        return action
    
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """
        Callable interface for policy evaluation.
        
        Allows policy to be used as: action = policy(obs)
        """
        return self.forward(obs)
    
    def get_parameters(self) -> np.ndarray:
        """
        Get parameters as flat vector θ.
        
        Returns:
            theta: Flat parameter vector (num_params,)
                  Format: [W.flatten(), b]
        """
        return np.concatenate([self.W.flatten(), self.b])
    
    def set_parameters(self, theta: np.ndarray) -> None:
        """
        Set parameters from flat vector θ.
        
        Args:
            theta: Flat parameter vector (num_params,)
        """
        assert theta.shape == (self.num_params,), \
            f"Expected {self.num_params} parameters, got {theta.shape[0]}"
        
        # Extract weight matrix
        w_size = self.action_dim * self.obs_dim
        w_flat = theta[:w_size]
        self.W = w_flat.reshape(self.action_dim, self.obs_dim)
        
        # Extract bias vector
        self.b = theta[w_size:]
    
    def clone(self) -> 'LinearPolicy':
        """
        Create deep copy of policy.
        
        Returns:
            copy: New LinearPolicy instance with identical parameters
        """
        # Create new policy with same dimensions
        copy = LinearPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            init_scale=0.0  # Will overwrite with clone's parameters
        )
        
        # Copy parameters
        copy.W = self.W.copy()
        copy.b = self.b.copy()
        
        return copy
    
    def mutate(self, sigma: float, rng: Optional[np.random.RandomState] = None) -> None:
        """
        Add Gaussian noise to parameters (in-place mutation).
        
        Implements:
            θ' = θ + N(0, σ²I)
        
        This is a standard mutation operator for evolutionary strategies.
        
        Args:
            sigma: Standard deviation of mutation noise
            rng: Random number generator (creates new one if None)
        """
        if rng is None:
            rng = np.random.RandomState()
        
        # Add Gaussian noise to weights
        self.W += rng.randn(*self.W.shape) * sigma
        
        # Add Gaussian noise to bias
        self.b += rng.randn(*self.b.shape) * sigma
    
    def get_num_parameters(self) -> int:
        """
        Get total number of parameters.
        
        Returns:
            count: Number of parameters in θ
        """
        return self.num_params
    
    def reset_parameters(self, init_scale: float = 0.1, seed: Optional[int] = None) -> None:
        """
        Reinitialize parameters randomly.
        
        Args:
            init_scale: Scale for random initialization
            seed: Random seed
        """
        rng = np.random.RandomState(seed)
        self.W = rng.randn(self.action_dim, self.obs_dim) * init_scale
        self.b = np.zeros(self.action_dim)


# Validation
if __name__ == "__main__":
    print("LinearPolicy - Continuous control policy")
    print("=" * 60)
    
    # Create policy
    obs_dim = 10  # e.g., 8 rays + 2 velocity
    action_dim = 2  # [ax, ay]
    
    policy = LinearPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        init_scale=0.1,
        seed=42
    )
    
    print(f"Policy configuration:")
    print(f"  Observation dimension: {policy.obs_dim}")
    print(f"  Action dimension: {policy.action_dim}")
    print(f"  Total parameters: {policy.get_num_parameters()}")
    print(f"  Weight matrix shape: {policy.W.shape}")
    print(f"  Bias vector shape: {policy.b.shape}")
    print()
    
    # Test forward pass
    obs = np.random.randn(obs_dim)
    action = policy(obs)
    
    print(f"Forward pass test:")
    print(f"  Input observation shape: {obs.shape}")
    print(f"  Output action shape: {action.shape}")
    print(f"  Output action: {action}")
    print(f"  Action range: [{action.min():.3f}, {action.max():.3f}]")
    assert -1.0 <= action.min() and action.max() <= 1.0, \
        "Action not in [-1, 1] range"
    print()
    
    # Test parameter extraction
    theta = policy.get_parameters()
    print(f"Parameter extraction:")
    print(f"  Flat parameter vector shape: {theta.shape}")
    print(f"  First 5 parameters: {theta[:5]}")
    print()
    
    # Test cloning
    policy_copy = policy.clone()
    theta_copy = policy_copy.get_parameters()
    
    print(f"Clone test:")
    print(f"  Parameters identical: {np.allclose(theta, theta_copy)}")
    print(f"  Different object: {policy is not policy_copy}")
    print()
    
    # Test mutation
    print(f"Mutation test:")
    theta_before = policy_copy.get_parameters().copy()
    policy_copy.mutate(sigma=0.1, rng=np.random.RandomState(123))
    theta_after = policy_copy.get_parameters()
    
    param_diff = np.linalg.norm(theta_after - theta_before)
    print(f"  Parameter L2 difference after mutation: {param_diff:.4f}")
    print(f"  Mean absolute change: {np.abs(theta_after - theta_before).mean():.4f}")
    print(f"  Max absolute change: {np.abs(theta_after - theta_before).max():.4f}")
    
    # Verify mutated policy produces different action
    action_after = policy_copy(obs)
    action_diff = np.linalg.norm(action_after - action)
    print(f"  Action difference after mutation: {action_diff:.4f}")
    print()
    
    # Test parameter setting
    print(f"Parameter setting test:")
    new_theta = np.random.randn(policy.num_params) * 0.05
    policy.set_parameters(new_theta)
    retrieved_theta = policy.get_parameters()
    print(f"  Set/get identical: {np.allclose(new_theta, retrieved_theta)}")
    print()
    
    # Test determinism
    print(f"Determinism test:")
    action1 = policy(obs)
    action2 = policy(obs)
    print(f"  Same observation → same action: {np.allclose(action1, action2)}")
    print()
    
    print("✓ LinearPolicy implementation complete")
    print("✓ Ready for evolutionary search and RL training")
    print("✓ Deterministic, parameterized, no internal learning")
