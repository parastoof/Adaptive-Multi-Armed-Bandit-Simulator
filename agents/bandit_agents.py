import numpy as np


class _BaseAgent:
    """Common scaffolding for all bandit agents."""

    def __init__(self, n_arms: int, seed=None):
        self.n_arms = int(n_arms)
        if self.n_arms <= 0:
            raise ValueError("n_arms must be positive")

        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def _reset_rng(self):
        self.rng = np.random.default_rng(self._seed)

    def select_action(self) -> int:
        raise NotImplementedError

    def update(self, action: int, reward: float) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class RandomAgent(_BaseAgent):

    def select_action(self) -> int:
        return int(self.rng.integers(self.n_arms))

    def update(self, action: int, reward: float) -> None:
        pass

    def reset(self) -> None:
        self._reset_rng()


class _EpsilonGreedyBase(_BaseAgent):

    def __init__(self, n_arms: int, epsilon: float, seed=None):
        super().__init__(n_arms, seed)

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")

        self.epsilon = float(epsilon)

    def _select_greedy(self) -> int:
        max_value = np.max(self.q_values)
        best_actions = np.flatnonzero(self.q_values == max_value)
        return int(self.rng.choice(best_actions))

    def select_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))
        return self._select_greedy()


class EpsilonGreedySampleAverage(_EpsilonGreedyBase):

    def __init__(self, n_arms: int = 5, epsilon: float = 0.1, seed=None):
        super().__init__(n_arms, epsilon, seed)
        self.reset()

    def reset(self) -> None:
        self.q_values = np.zeros(self.n_arms, dtype=float)
        self.action_counts = np.zeros(self.n_arms, dtype=int)
        self._reset_rng()

    def update(self, action: int, reward: float) -> None:
        self.action_counts[action] += 1
        n = self.action_counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / n


class EpsilonGreedyConstantStep(_EpsilonGreedyBase):

    def __init__(
        self,
        n_arms: int = 5,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        seed=None,
    ):
        super().__init__(n_arms, epsilon, seed)

        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")

        self.alpha = float(alpha)
        self.reset()

    def reset(self) -> None:
        self.q_values = np.zeros(self.n_arms, dtype=float)
        self._reset_rng()

    def update(self, action: int, reward: float) -> None:
        self.q_values[action] += self.alpha * (
            reward - self.q_values[action]
        )