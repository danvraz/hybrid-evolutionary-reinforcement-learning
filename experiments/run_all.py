"""
experiments/run_all.py

Controlled experimental comparison of training methods.

Runs three training paradigms under identical conditions:
1. Pure Evolution (ES)
2. Pure Reinforcement Learning (Actor-Critic)
3. Hybrid (Evolution → RL fine-tuning)
"""

import sys
import os
import numpy as np
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_shared_config():
    return {
        'maze_width': 15,
        'maze_height': 15,
        'cell_size': 1.0,
        'agent_radius': 0.2,
        'agent_max_speed': 2.0,
        'agent_num_rays': 8,
        'agent_ray_max_distance': 5.0,
        'agent_dt': 0.05,
        'max_steps': 500,
        'goal_threshold': 2.0,
        'step_penalty': -0.0005,
        'goal_reward': 100.0,
        'distance_reward_scale': 15.0,
        'include_velocity_obs': True,
        'policy_init_scale': 0.1
    }


def create_components(config: Dict, env_seed: int, policy_seed: int):
    from env.maze_env import MazeEnv
    from sim.agent import Agent
    from policy.linear_policy import LinearPolicy
    from eval.rollout import RolloutEvaluator

    env = MazeEnv(
        width=config['maze_width'],
        height=config['maze_height'],
        cell_size=config['cell_size'],
        seed=env_seed
    )

    agent = Agent(
        position=(1.0, 1.0),
        radius=config['agent_radius'],
        max_speed=config['agent_max_speed'],
        num_rays=config['agent_num_rays'],
        ray_max_distance=config['agent_ray_max_distance'],
        dt=config['agent_dt']
    )

    policy_template = LinearPolicy(
        obs_dim=agent.observation_dim,
        action_dim=agent.action_dim,
        init_scale=config['policy_init_scale'],
        seed=policy_seed
    )

    evaluator = RolloutEvaluator(
        max_steps=config['max_steps'],
        goal_threshold=config['goal_threshold'],
        step_penalty=config['step_penalty'],
        goal_reward=config['goal_reward'],
        distance_reward_scale=config['distance_reward_scale'],
        include_velocity_obs=config['include_velocity_obs']
    )

    return {
        'env': env,
        'agent': agent,
        'policy_template': policy_template,
        'evaluator': evaluator
    }


def run_evolution_baseline(config, num_generations, population_size, seed):
    from algo.evolution import EvolutionaryStrategy

    print("=" * 70)
    print("EXPERIMENT 1: PURE EVOLUTION")
    print("=" * 70)

    components = create_components(config, env_seed=seed, policy_seed=seed)
    policy = components['policy_template'].clone()

    evolution = EvolutionaryStrategy(
        policy_template=policy,
        evaluator=components['evaluator'],
        env=components['env'],
        agent=components['agent'],
        population_size=population_size,
        elite_frac=0.2,
        mutation_sigma=0.1,
        num_eval_episodes=1,
        seed=seed
    )

    start = time.time()
    evo_results = evolution.train(num_generations=num_generations, verbose=True)
    total_time = time.time() - start

    eval_rewards, eval_success = [], []
    print(f"\nEvaluating final policy over 10 episodes...")
    for ep in range(10):
        m = components['evaluator'].evaluate(
            components['env'], components['agent'], evo_results['best_policy'],
            seed=seed + 1000 + ep
        )
        eval_rewards.append(m['cumulative_reward'])
        eval_success.append(m['success'])

    results = {
        'method': 'Evolution',
        'final_mean_reward': np.mean(eval_rewards),
        'final_std_reward': np.std(eval_rewards),
        'final_success_rate': np.mean(eval_success),
        'total_time': total_time,
        'total_env_steps': num_generations * population_size * config['max_steps'],
        'history': evo_results['history'],
        'best_policy': evo_results['best_policy']
    }
    
    print(f"\nEvolution Results:")
    print(f"  Final reward: {results['final_mean_reward']:.2f} ± {results['final_std_reward']:.2f}")
    print(f"  Success rate: {results['final_success_rate']:.2%}")
    print(f"  Total time: {results['total_time']:.1f}s")
    print(f"  Total env steps: {results['total_env_steps']:,}")
    
    return results


def run_rl_baseline(config, num_iterations, episodes_per_iteration, seed):
    from algo.rl import ActorCritic
    from policy.actor_critic_policy import ActorCriticPolicy

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: PURE RL (ACTOR-CRITIC)")
    print("=" * 70)

    components = create_components(config, env_seed=seed, policy_seed=seed)

    policy = ActorCriticPolicy(
        obs_dim=components['agent'].observation_dim,
        action_dim=components['agent'].action_dim,
        init_scale=config['policy_init_scale'],
        seed=seed
    )

    rl = ActorCritic(
        policy=policy,
        evaluator=components['evaluator'],
        env=components['env'],
        agent=components['agent'],
        actor_lr=0.001,
        critic_lr=0.005,
        gamma=0.99,
        gae_lambda=0.95,
        episodes_per_iteration=episodes_per_iteration,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        seed=seed
    )

    start = time.time()
    train_results = rl.train(num_iterations=num_iterations, verbose=True)
    total_time = time.time() - start

    eval_rewards, eval_success = [], []
    print(f"\nEvaluating final policy over 10 episodes...")
    for ep in range(10):
        m = components['evaluator'].evaluate(
            components['env'], components['agent'], train_results['policy'],
            seed=seed + 2000 + ep
        )
        eval_rewards.append(m['cumulative_reward'])
        eval_success.append(m['success'])

    results = {
        'method': 'RL (Actor-Critic)',
        'final_mean_reward': np.mean(eval_rewards),
        'final_std_reward': np.std(eval_rewards),
        'final_success_rate': np.mean(eval_success),
        'total_time': total_time,
        'total_env_steps': num_iterations * episodes_per_iteration * config['max_steps'],
        'history': train_results['history'],
        'best_policy': train_results['policy']
    }
    
    print(f"\nRL Results:")
    print(f"  Final reward: {results['final_mean_reward']:.2f} ± {results['final_std_reward']:.2f}")
    print(f"  Success rate: {results['final_success_rate']:.2%}")
    print(f"  Total time: {results['total_time']:.1f}s")
    print(f"  Total env steps: {results['total_env_steps']:,}")
    
    return results


def run_hybrid_baseline(
    config,
    num_evolution_generations,
    num_rl_iterations,
    population_size,
    episodes_per_iteration,
    seed_evo,
    seed_rl
):
    from algo.evolution import EvolutionaryStrategy
    from algo.rl import ActorCritic
    from algo.hybrid import HybridEvolutionRL
    from policy.actor_critic_policy import ActorCriticPolicy

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: HYBRID (EVOLUTION → ACTOR-CRITIC)")
    print("=" * 70)

    components = create_components(config, env_seed=seed_evo, policy_seed=seed_evo)

    evo_policy = components['policy_template'].clone()

    evolution = EvolutionaryStrategy(
        policy_template=evo_policy,
        evaluator=components['evaluator'],
        env=components['env'],
        agent=components['agent'],
        population_size=population_size,
        elite_frac=0.2,
        mutation_sigma=0.1,
        num_eval_episodes=1,
        seed=seed_evo
    )

    ac_policy = ActorCriticPolicy(
        obs_dim=components['agent'].observation_dim,
        action_dim=components['agent'].action_dim,
        init_scale=0.01,
        seed=seed_rl
    )

    rl = ActorCritic(
        policy=ac_policy,
        evaluator=components['evaluator'],
        env=components['env'],
        agent=components['agent'],
        actor_lr=0.001,
        critic_lr=0.005,
        gamma=0.99,
        gae_lambda=0.95,
        episodes_per_iteration=episodes_per_iteration,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        seed=seed_rl
    )

    hybrid = HybridEvolutionRL(evolution, rl, verbose=True)

    start = time.time()
    results = hybrid.train(num_evolution_generations, num_rl_iterations)
    total_time = time.time() - start

    eval_rewards, eval_success = [], []
    print(f"\nEvaluating final policy over 10 episodes...")
    for ep in range(10):
        m = components['evaluator'].evaluate(
            components['env'], components['agent'], results['final_policy'],
            seed=seed_rl + 3000 + ep
        )
        eval_rewards.append(m['cumulative_reward'])
        eval_success.append(m['success'])

    hybrid_results = {
        'method': 'Hybrid (Evo→AC)',
        'final_mean_reward': np.mean(eval_rewards),
        'final_std_reward': np.std(eval_rewards),
        'final_success_rate': np.mean(eval_success),
        'total_time': total_time,
        'total_env_steps': results['total_env_steps'],
        'history': results['combined_history'],
        'best_policy': results['final_policy']
    }
    
    print(f"\nHybrid Results:")
    print(f"  Final reward: {hybrid_results['final_mean_reward']:.2f} ± {hybrid_results['final_std_reward']:.2f}")
    print(f"  Success rate: {hybrid_results['final_success_rate']:.2%}")
    print(f"  Total time: {hybrid_results['total_time']:.1f}s")
    print(f"  Total env steps: {hybrid_results['total_env_steps']:,}")
    
    return hybrid_results


def print_comparison_table(results: List[Dict]):
    print("\n")
    print("=" * 70)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("=" * 70)
    print()
    
    print(f"{'Method':<20} | {'Reward':<15} | {'Success':<10} | {'Time (s)':<10} | {'Env Steps':<12}")
    print("-" * 70)
    
    for result in results:
        method = result['method']
        reward = f"{result['final_mean_reward']:6.2f} ± {result['final_std_reward']:4.2f}"
        success = f"{result['final_success_rate']:5.1%}"
        time_str = f"{result['total_time']:8.1f}"
        steps = f"{result['total_env_steps']:>10,}"
        
        print(f"{method:<20} | {reward:<15} | {success:<10} | {time_str:<10} | {steps:<12}")
    
    print("=" * 70)
    print()
    
    best_reward_idx = np.argmax([r['final_mean_reward'] for r in results])
    best_success_idx = np.argmax([r['final_success_rate'] for r in results])
    
    print("Key Findings:")
    print(f"  Best reward: {results[best_reward_idx]['method']}")
    print(f"  Best success rate: {results[best_success_idx]['method']}")
    print()


def run_all_experiments(
    num_evo_generations=50,
    num_rl_iterations=100,
    num_hybrid_evo_gen=25,
    num_hybrid_rl_iter=50,
    population_size=50,
    episodes_per_iteration=10
):
    print("EXPERIMENTAL COMPARISON: EVOLUTION vs RL vs HYBRID")
    print("Research Question: Does evolutionary pre-training improve RL?")
    print()
    
    config = get_shared_config()
    
    print("Shared configuration:")
    print(f"  Maze: {config['maze_width']}x{config['maze_height']}")
    print(f"  Agent: {config['agent_num_rays']} rays, radius={config['agent_radius']}")
    print(f"  Max steps per episode: {config['max_steps']}")
    print(f"  Policy init scale: {config['policy_init_scale']}")
    print()
    
    results = []
    
    # 1. Pure Evolution
    evo_results = run_evolution_baseline(
        config=config,
        num_generations=num_evo_generations,
        population_size=population_size,
        seed=42
    )
    results.append(evo_results)
    
    # 2. Pure RL
    rl_results = run_rl_baseline(
        config=config,
        num_iterations=num_rl_iterations,
        episodes_per_iteration=episodes_per_iteration,
        seed=43
    )
    results.append(rl_results)
    
    # 3. Hybrid
    hybrid_results = run_hybrid_baseline(
        config=config,
        num_evolution_generations=num_hybrid_evo_gen,
        num_rl_iterations=num_hybrid_rl_iter,
        population_size=population_size,
        episodes_per_iteration=episodes_per_iteration,
        seed_evo=44,
        seed_rl=45
    )
    results.append(hybrid_results)
    
    print_comparison_table(results)
    
    return results


if __name__ == "__main__":
    print("Starting experimental runs...")
    print()
    
    config = {
        'num_evo_generations': 10,
        'num_rl_iterations': 10,
        'num_hybrid_evo_gen': 5,
        'num_hybrid_rl_iter': 5,
        'population_size': 20,
        'episodes_per_iteration': 5
    }
    
    print("Experimental configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    results = run_all_experiments(**config)
    
    print("All experiments complete!")
