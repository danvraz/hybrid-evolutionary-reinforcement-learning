Hybrid Evolutionary + Reinforcement Learning for Navigation

This project investigates whether evolutionary pre-training can improve reinforcement learning performance in sparse-reward, continuous-control navigation tasks.

We compare three training paradigms under identical conditions:

Pure Evolutionary Strategy (ES)

Pure Reinforcement Learning (Actor-Critic)

Hybrid: Evolution → Actor-Critic fine-tuning

Motivation

Sparse-reward environments are notoriously difficult for reinforcement learning due to poor exploration and high gradient variance.

Evolutionary methods, while sample-inefficient, excel at global exploration.

This project asks:

Can evolution discover useful sensorimotor priors that accelerate subsequent RL training?

Environment

2D maze navigation

Continuous action space (velocity control)

Partial observability via range sensors

Sparse terminal reward upon reaching the goal

Methods
1. Evolutionary Strategy

Population-based optimization

Gaussian mutation

Fitness based on episodic return

2. Actor-Critic (A2C-style)

Stochastic Gaussian policy

Learned value function (critic)

Advantage estimation (GAE)

Entropy regularization for exploration

3. Hybrid (Evolution → RL)

Run evolution first

Extract best evolved policy

Initialize Actor-Critic actor weights

Fine-tune via gradient-based RL

Results (Representative Run)
Method	Final Reward	Success Rate	Env Steps
Evolution	-2.81 ± 48.19	0%	100,000
RL (Actor-Critic)	5.54 ± 43.57	0%	25,000
Hybrid (Evo→AC)	31.60 ± 93.30	10%	62,500

Key finding:
Hybrid training achieves higher reward and non-zero success, while standalone methods fail under the same budget


How to Run
python experiments/run_all.py


Hyperparameters can be adjusted in experiments/run_all.py.


Project Structure
algo/        # Evolution, Actor-Critic, Hybrid pipeline
env/         # Maze environment
policy/      # Linear & Actor-Critic policies
sim/         # Agent physics and sensors
eval/        # Rollout evaluation
experiments/ # Controlled experiment runner
Research Status

This is an ongoing research project.

Planned extensions:

Multi-seed statistical evaluation

Learning curve plots

Larger mazes and harder tasks

Paper submission draft


Author

Dan
