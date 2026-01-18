"""
algo/hybrid.py

Hybrid Evolutionary + Reinforcement Learning pipeline.
"""

import numpy as np
from typing import Dict
import time


class HybridEvolutionRL:
    def __init__(self, evolution_trainer, rl_trainer, verbose: bool = True):
        self.evolution = evolution_trainer
        self.rl = rl_trainer
        self.verbose = verbose

        assert self.evolution.policy_template.obs_dim == self.rl.policy.obs_dim
        assert self.evolution.policy_template.action_dim == self.rl.policy.action_dim

        self.history = {
            'stage': [],
            'iteration': [],
            'best_fitness': [],
            'mean_fitness': [],
            'mean_reward': [],
            'success_rate': [],
            'total_env_steps': [],
        }

        self.total_env_steps = 0
        self.evolution_complete = False
        self.rl_complete = False

    def train(self, num_evolution_generations: int, num_rl_iterations: int) -> Dict:
        if self.verbose:
            print("=" * 70)
            print("HYBRID EVOLUTION + RL TRAINING PIPELINE")
            print("=" * 70)
            print(f"Stage 1: Evolution for {num_evolution_generations} generations")
            print(f"Stage 2: RL fine-tuning for {num_rl_iterations} iterations")
            print("=" * 70)
            print()

        total_start_time = time.time()

        # STAGE 1: EVOLUTION
        if self.verbose:
            print("STAGE 1: EVOLUTIONARY SEARCH")
            print("-" * 70)

        evo_start_time = time.time()

        evo_results = self.evolution.train(
            num_generations=num_evolution_generations,
            verbose=self.verbose
        )

        evo_time = time.time() - evo_start_time
        best_evolved_policy = evo_results['best_policy']

        evo_env_steps = (
            num_evolution_generations *
            self.evolution.population_size *
            self.evolution.num_eval_episodes *
            self.evolution.evaluator.max_steps
        )
        self.total_env_steps += evo_env_steps

        evo_history = evo_results['history']
        for i in range(len(evo_history['generation'])):
            self.history['stage'].append('evolution')
            self.history['iteration'].append(evo_history['generation'][i])
            self.history['best_fitness'].append(evo_history['best_fitness'][i])
            self.history['mean_fitness'].append(evo_history['best_fitness'][i])
            self.history['mean_reward'].append(None)
            self.history['success_rate'].append(evo_history['success_rate'][i])
            self.history['total_env_steps'].append(
                self.total_env_steps * (i + 1) / num_evolution_generations
            )

        if self.verbose:
            print()
            print(f"Evolution complete: {evo_time:.1f}s")
            print(f"Best evolved fitness: {evo_results['best_fitness']:.2f}")
            print(f"Environment steps: {evo_env_steps:,}")
            print()

        self.evolution_complete = True

        # STAGE 2: RL FINE-TUNING
        if self.verbose:
            print("STAGE 2: RL FINE-TUNING")
            print("-" * 70)

        rl_start_time = time.time()

        # Initialize Actor-Critic from evolved policy
        from policy.actor_critic_policy import ActorCriticPolicy

        evolved_params = best_evolved_policy.get_parameters()

        # Get RL policy (should already be ActorCriticPolicy)
        # Initialize its actor from evolved linear policy
        w_size = best_evolved_policy.action_dim * best_evolved_policy.obs_dim
        self.rl.policy.W_actor[:] = evolved_params[:w_size].reshape(
            best_evolved_policy.action_dim,
            best_evolved_policy.obs_dim
        )
        self.rl.policy.b_actor[:] = evolved_params[w_size:w_size + best_evolved_policy.action_dim]

        if self.verbose:
            print("Initialized Actor-Critic actor from evolved policy")
            print(f"Actor parameter norm: {np.linalg.norm(self.rl.policy.W_actor):.4f}")
            print()

        rl_results = self.rl.train(
            num_iterations=num_rl_iterations,
            verbose=self.verbose
        )

        rl_time = time.time() - rl_start_time
        final_policy = rl_results['policy']

        rl_env_steps = (
            num_rl_iterations *
            self.rl.episodes_per_iteration *
            self.rl.evaluator.max_steps
        )
        self.total_env_steps += rl_env_steps

        rl_history = rl_results['history']
        for i in range(len(rl_history['iteration'])):
            self.history['stage'].append('rl')
            self.history['iteration'].append(rl_history['iteration'][i])
            self.history['best_fitness'].append(None)
            self.history['mean_fitness'].append(None)
            self.history['mean_reward'].append(rl_history['mean_reward'][i])
            self.history['success_rate'].append(rl_history['success_rate'][i])
            self.history['total_env_steps'].append(
                evo_env_steps + rl_env_steps * (i + 1) / num_rl_iterations
            )

        self.rl_complete = True

        total_time = time.time() - total_start_time

        if self.verbose:
            print()
            print(f"RL fine-tuning complete: {rl_time:.1f}s")
            print(f"Final mean reward: {rl_history['mean_reward'][-1]:.2f}")
            print(f"Environment steps: {rl_env_steps:,}")
            print()
            print("=" * 70)
            print("HYBRID TRAINING COMPLETE")
            print("=" * 70)
            print(f"Total time: {total_time:.1f}s")
            print(f"  Evolution: {evo_time:.1f}s")
            print(f"  RL: {rl_time:.1f}s")
            print(f"Total environment steps: {self.total_env_steps:,}")
            print("=" * 70)
            print()

        return {
            'final_policy': final_policy,
            'best_evolved_policy': best_evolved_policy,
            'evolution_history': evo_history,
            'rl_history': rl_history,
            'combined_history': self.history,
            'total_time': total_time,
            'evolution_time': evo_time,
            'rl_time': rl_time,
            'total_env_steps': self.total_env_steps
        }

    def get_policy(self):
        if not self.rl_complete:
            raise RuntimeError("Training not complete. Call train() first.")
        return self.rl.get_policy()

    def get_history(self):
        return self.history

    def get_evolution_policy(self):
        if not self.evolution_complete:
            raise RuntimeError("Evolution stage not complete. Call train() first.")
        return self.evolution.get_best_policy()
