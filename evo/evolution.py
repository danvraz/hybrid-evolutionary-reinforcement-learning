import numpy as np
from sim.policy import Policy
from evo.fitness import EpisodeRunner

class Evolution:
    def __init__(self, env, population_size=20, elite_frac=0.2, mutation_sigma=0.15):
        self.env = env
        self.pop_size = population_size
        self.elite_frac = elite_frac
        self.sigma = mutation_sigma

        self.runner = EpisodeRunner(env)
        self.population = [Policy() for _ in range(population_size)]

    def evaluate_population(self):
        results = []
        for p in self.population:
            info = self.runner.run(p)
            results.append((info["distance"], p))
        results.sort(key=lambda x: x[0])
        return results

    def next_generation(self, ranked):
        elite_count = int(self.elite_frac * self.pop_size)
        elites = [p.clone() for _, p in ranked[:elite_count]]

        new_pop = elites.copy()
        while len(new_pop) < self.pop_size:
            parent = np.random.choice(elites)
            child = parent.clone()
            child.mutate(self.sigma)
            new_pop.append(child)

        self.population = new_pop

    def run(self, generations=10):
        history = []
        for g in range(generations):
            ranked = self.evaluate_population()
            best_dist = ranked[0][0]
            history.append(best_dist)
            self.next_generation(ranked)
            print(f"Gen {g}: best distance = {best_dist:.3f}")
        return history

