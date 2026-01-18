import numpy as np

class Policy:
    def __init__(self, obs_dim=8, action_dim=2, hidden_dim=16):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.W1 = np.random.randn(hidden_dim, obs_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(action_dim, hidden_dim) * 0.5
        self.b2 = np.zeros(action_dim)

    def forward(self, obs):
        h = np.tanh(self.W1 @ obs + self.b1)
        action = np.tanh(self.W2 @ h + self.b2)
        return action

    def clone(self):
        p = Policy(self.obs_dim, self.action_dim)
        p.W1 = self.W1.copy()
        p.b1 = self.b1.copy()
        p.W2 = self.W2.copy()
        p.b2 = self.b2.copy()
        return p

    def mutate(self, sigma=0.1):
        for param in [self.W1, self.b1, self.W2, self.b2]:
            param += np.random.randn(*param.shape) * sigma

