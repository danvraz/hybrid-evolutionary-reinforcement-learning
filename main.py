from env.maze_env import MazeEnv
from evo.evolution import Evolution

if __name__ == "__main__":
    env = MazeEnv()
    evo = Evolution(env)
    history = evo.run(generations=20)
    print("Final:", history[-1])

