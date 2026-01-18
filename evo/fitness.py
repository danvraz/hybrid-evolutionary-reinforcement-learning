import numpy as np
from sim.policy import Policy

class EpisodeRunner:
    def __init__(self, env, max_steps=1000):
        self.env = env
        self.max_steps = max_steps

    def run(self, policy):
        obs = self.env.reset()
        total_reward = 0.0

        for step in range(self.max_steps):
            action = policy.forward(obs)
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done:
                break

        dist = np.linalg.norm(self.env.agent_pos - self.env.goal)
        return {
            "reward": total_reward,
            "distance": dist,
            "steps": step
        }
