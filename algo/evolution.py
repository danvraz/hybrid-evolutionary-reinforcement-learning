"""
algo/evolution.py

Pure Evolutionary Strategies (ES) for policy optimization.

Implements a simple (μ, λ)-ES with elitism:
- Population of policies parameterized by θ
- Gaussian mutation operator
- Fitness-based selection
- Elite preservation

This is a gradient-free baseline that will be compared against:
- Reinforcement learning (RL-only)
- Hybrid evolution → RL

Algorithm:
    1. Initialize population randomly
    2. For each generation:
        a. Clone and mutate population
        b. Evaluate all policies (fitness = episode reward)
        c. Rank by fitness
        d. Select top elite_frac as parents
        e. Generate new population from elite mutations
    3. Return best policy found

This module does NOT:
- Use gradients
- Modify environment or agent
- Implement RL algorithms
"""

import numpy as np
from typing import Dict, List, Optional
import time


class EvolutionaryStrategy:
    """
    Evolutionary strategy for policy optimization.
    
    Uses fitness-based selection with Gaussian mutation to evolve
    a population of policies over generations.
    
    Hyperparameters:
    - population_size: Number of policies per generation
    - elite_frac: Fraction of population to preserve as elites
    - mutation_sigma: Standard deviation of parameter mutations
    """
    
    def __init__(
        self,
        policy_template,  # LinearPolicy instance (used as template)
        evaluator,  # RolloutEvaluator instance
        env,  # MazeEnv instance
        agent,  # Agent instance
        population_size: int = 50,
        elite_frac: float = 0.2,
        mutation_sigma: float = 0.1,
        num_eval_episodes: int = 1,
        seed: Optional[int] = None
    ):
        """
        Initialize evolutionary strategy.
        
        Args:
            policy_template: Policy instance to clone for population
            evaluator: Evaluator for fitness computation
            env: Environment for evaluation
            agent: Agent for evaluation
            population_size: Number of policies in population
            elite_frac: Fraction to preserve as elites (0 < elite_frac < 1)
            mutation_sigma: Mutation strength (Gaussian std dev)
            num_eval_episodes: Episodes to average for fitness
            seed: Random seed for reproducibility
        """
        assert 0 < elite_frac < 1, "elite_frac must be in (0, 1)"
        assert population_size > 0, "population_size must be positive"
        
        self.policy_template = policy_template
        self.evaluator = evaluator
        self.env = env
        self.agent = agent
        
        self.population_size = population_size
        self.elite_size = max(1, int(population_size * elite_frac))
        self.mutation_sigma = mutation_sigma
        self.num_eval_episodes = num_eval_episodes
        
        # Random number generator
        self.rng = np.random.RandomState(seed)
        
        # Initialize population
        self.population = self._initialize_population()
        
        # Tracking
        self.generation = 0
        self.best_fitness = -np.inf
        self.best_policy = None
        
        # History for logging
        self.history = {
            'generation': [],
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
            'success_rate': [],
            'best_steps': [],
        }
    
    def _initialize_population(self) -> List:
        """
        Create initial population of random policies.
        
        Returns:
            population: List of policy instances
        """
        population = []
        
        for i in range(self.population_size):
            # Clone template and randomize parameters
            policy = self.policy_template.clone()
            policy.reset_parameters(
                init_scale=0.1,
                seed=self.rng.randint(0, 1000000)
            )
            population.append(policy)
        
        return population
    
    def _evaluate_policy(self, policy) -> Dict:
        """
        Evaluate single policy over multiple episodes.
        
        Args:
            policy: Policy to evaluate
        
        Returns:
            metrics: Averaged metrics over episodes
        """
        episode_rewards = []
        episode_successes = []
        episode_steps = []
        
        for ep in range(self.num_eval_episodes):
            # Generate unique seed for this episode
            ep_seed = self.rng.randint(0, 1000000)
            
            # Run evaluation
            metrics = self.evaluator.evaluate(
                env=self.env,
                agent=self.agent,
                policy=policy,
                seed=ep_seed
            )
            
            episode_rewards.append(metrics['cumulative_reward'])
            episode_successes.append(metrics['success'])
            episode_steps.append(metrics['steps'])
        
        # Average over episodes
        return {
            'fitness': np.mean(episode_rewards),
            'success_rate': np.mean(episode_successes),
            'mean_steps': np.mean(episode_steps)
        }
    
    def _evaluate_population(self) -> List[Dict]:
        """
        Evaluate all policies in population.
        
        Returns:
            results: List of metric dictionaries (one per policy)
        """
        results = []
        
        for i, policy in enumerate(self.population):
            metrics = self._evaluate_policy(policy)
            results.append(metrics)
        
        return results
    
    def step(self) -> Dict:
        """
        Execute one generation of evolution.
        
        Steps:
        1. Evaluate current population
        2. Rank by fitness
        3. Select elites
        4. Generate offspring via mutation
        5. Form new population
        
        Returns:
            summary: Generation statistics
        """
        # Evaluate population
        eval_results = self._evaluate_population()
        
        # Extract fitness values
        fitness_values = np.array([r['fitness'] for r in eval_results])
        success_rates = np.array([r['success_rate'] for r in eval_results])
        steps_taken = np.array([r['mean_steps'] for r in eval_results])
        
        # Rank policies by fitness (descending)
        sorted_indices = np.argsort(fitness_values)[::-1]
        
        # Select elites
        elite_indices = sorted_indices[:self.elite_size]
        elites = [self.population[i] for i in elite_indices]
        
        # Track best policy
        best_idx = sorted_indices[0]
        current_best_fitness = fitness_values[best_idx]
        
        if current_best_fitness > self.best_fitness:
            self.best_fitness = current_best_fitness
            self.best_policy = self.population[best_idx].clone()
        
        # Generate new population
        new_population = []
        
        # Preserve elites (no mutation)
        for elite in elites:
            new_population.append(elite.clone())
        
        # Fill rest with mutated elites
        while len(new_population) < self.population_size:
            # Sample random elite
            parent_idx = self.rng.randint(len(elites))
            parent = elites[parent_idx]
            
            # Clone and mutate
            offspring = parent.clone()
            offspring.mutate(sigma=self.mutation_sigma, rng=self.rng)
            
            new_population.append(offspring)
        
        # Update population
        self.population = new_population
        self.generation += 1
        
        # Compute statistics
        summary = {
            'generation': self.generation,
            'best_fitness': fitness_values[best_idx],
            'mean_fitness': np.mean(fitness_values),
            'std_fitness': np.std(fitness_values),
            'success_rate': success_rates[best_idx],
            'mean_success_rate': np.mean(success_rates),
            'best_steps': steps_taken[best_idx]
        }
        
        # Log to history
        for key in self.history.keys():
            if key in summary:
                self.history[key].append(summary[key])
            elif key == 'success_rate':
                self.history[key].append(summary['mean_success_rate'])
        
        return summary
    
    def train(self, num_generations: int, verbose: bool = True) -> Dict:
        """
        Run evolution for specified number of generations.
        
        Args:
            num_generations: Number of generations to evolve
            verbose: Whether to print progress
        
        Returns:
            final_summary: Training summary with best policy
        """
        if verbose:
            print(f"Starting evolution for {num_generations} generations")
            print(f"Population size: {self.population_size}")
            print(f"Elite size: {self.elite_size}")
            print(f"Mutation sigma: {self.mutation_sigma}")
            print("=" * 60)
        
        start_time = time.time()
        
        for gen in range(num_generations):
            gen_start = time.time()
            
            # Execute one generation
            summary = self.step()
            
            gen_time = time.time() - gen_start
            
            # Print progress
            if verbose:
                print(
                    f"Gen {summary['generation']:3d} | "
                    f"Best: {summary['best_fitness']:7.2f} | "
                    f"Mean: {summary['mean_fitness']:7.2f} ± {summary['std_fitness']:5.2f} | "
                    f"Success: {summary['mean_success_rate']:.2%} | "
                    f"Time: {gen_time:.1f}s"
                )
        
        total_time = time.time() - start_time
        
        if verbose:
            print("=" * 60)
            print(f"Evolution complete in {total_time:.1f}s")
            print(f"Best fitness achieved: {self.best_fitness:.2f}")
        
        return {
            'best_policy': self.best_policy,
            'best_fitness': self.best_fitness,
            'history': self.history,
            'total_time': total_time
        }
    
    def get_best_policy(self):
        """
        Get best policy found so far.
        
        Returns:
            policy: Best policy (cloned)
        """
        if self.best_policy is None:
            raise RuntimeError("No policy evaluated yet. Run train() first.")
        return self.best_policy.clone()
    
    def get_history(self) -> Dict:
        """
        Get training history.
        
        Returns:
            history: Dictionary of logged metrics over generations
        """
        return self.history


# Validation and demo
if __name__ == "__main__":
    print("Evolutionary Strategy - Pure ES baseline")
    print("=" * 60)
    print()
    
    # Mock imports for standalone testing
    # In actual use, these would import from other modules
    
    class MockPolicy:
        def __init__(self):
            self.obs_dim = 10
            self.action_dim = 2
            self.theta = np.random.randn(22) * 0.1
        
        def clone(self):
            policy = MockPolicy()
            policy.theta = self.theta.copy()
            return policy
        
        def reset_parameters(self, init_scale=0.1, seed=None):
            rng = np.random.RandomState(seed)
            self.theta = rng.randn(22) * init_scale
        
        def mutate(self, sigma, rng):
            self.theta += rng.randn(22) * sigma
        
        def __call__(self, obs):
            # Simple linear policy
            return np.tanh(self.theta[:2])
    
    class MockEvaluator:
        def evaluate(self, env, agent, policy, seed=None):
            # Mock evaluation - random fitness
            rng = np.random.RandomState(seed)
            fitness = rng.randn() * 10 + np.sum(np.abs(policy.theta)) * 0.1
            success = fitness > 5.0
            return {
                'cumulative_reward': fitness,
                'success': success,
                'steps': 100
            }
    
    class MockEnv:
        pass
    
    class MockAgent:
        pass
    
    print("Creating mock components...")
    policy_template = MockPolicy()
    evaluator = MockEvaluator()
    env = MockEnv()
    agent = MockAgent()
    
    print("Initializing evolutionary strategy...")
    es = EvolutionaryStrategy(
        policy_template=policy_template,
        evaluator=evaluator,
        env=env,
        agent=agent,
        population_size=20,
        elite_frac=0.2,
        mutation_sigma=0.1,
        num_eval_episodes=1,
        seed=42
    )
    
    print(f"Population size: {es.population_size}")
    print(f"Elite size: {es.elite_size}")
    print()
    
    print("Running evolution for 10 generations...")
    print()
    
    results = es.train(num_generations=10, verbose=True)
    
    print()
    print("Final results:")
    print(f"  Best fitness: {results['best_fitness']:.2f}")
    print(f"  Total time: {results['total_time']:.2f}s")
    print(f"  Generations: {len(results['history']['generation'])}")
    print()
    
    # Show learning curve
    print("Learning curve (best fitness per generation):")
    for gen, fitness in enumerate(results['history']['best_fitness'][:10]):
        bar = '#' * int(fitness / 2)
        print(f"  Gen {gen:2d}: {fitness:6.2f} {bar}")
    
    print()
    print("✓ Evolution implementation complete")
    print("✓ Deterministic, gradient-free, population-based")
    print("✓ Ready for comparison with RL and hybrid methods")
