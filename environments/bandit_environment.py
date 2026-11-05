import numpy as np


DEFAULT_PROBABILITIES = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)
NON_STATIONARY_CHANGE_STEP = 500  # تغییر قبل از گام 501 رخ می‌دهد


class BanditEnvironment:

    def __init__(
        self,
        env_type: str,
        seed=None,
        reward_uniforms=None,
        initial_probabilities=None,
    ):
        if env_type not in ("stationary", "non_stationary"):
            raise ValueError(f"Unknown environment type: {env_type}")

        self.env_type = env_type
        self.seed = seed

        base = (
            np.asarray(initial_probabilities, dtype=float)
            if initial_probabilities is not None
            else DEFAULT_PROBABILITIES
        )

        if np.any((base < 0) | (base > 1)):
            raise ValueError("Probabilities must be in [0, 1]")

        self.initial_probabilities = base.copy()
        self.n_arms = len(self.initial_probabilities)
        self.change_step = NON_STATIONARY_CHANGE_STEP

        if reward_uniforms is not None:
            reward_uniforms = np.asarray(reward_uniforms, dtype=float)

            if reward_uniforms.ndim != 2:
                raise ValueError("reward_uniforms must be 2D")
            if reward_uniforms.shape[1] != self.n_arms:
                raise ValueError("reward_uniforms has wrong number of arms")
            if np.any((reward_uniforms < 0) | (reward_uniforms >= 1)):
                raise ValueError("reward_uniforms values must be in [0, 1)")

            self.reward_uniforms = reward_uniforms
        else:
            self.reward_uniforms = None

        self.rng = np.random.default_rng(seed)
        self.current_step = 0  # تعداد گام‌های انجام‌شده

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self) -> int:
        self.current_step = 0
        self.rng = np.random.default_rng(self.seed)
        return self.current_step

    # ------------------------------------------------------------------
    # Probabilities
    # ------------------------------------------------------------------
    def get_probabilities(self):
        """
        Return the true reward probabilities for the CURRENT step.

        Called BEFORE the step is taken (i.e. when current_step == t-1),
        this returns the probabilities that will apply to step t.
        """
        if self.env_type == "stationary":
            return self.initial_probabilities.copy()

        # Non-stationary: change happens BEFORE step `change_step + 1`
        if self.current_step < self.change_step:
            return self.initial_probabilities.copy()

        return self.initial_probabilities[::-1].copy()

    def optimal_value(self) -> float:
        return float(np.max(self.get_probabilities()))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action: int) -> int:
        if not 0 <= action < self.n_arms:
            raise ValueError(f"Invalid action: {action}")

        probabilities = self.get_probabilities()  # ← احتمالات گام جاری

        if self.reward_uniforms is not None:
            if self.current_step >= len(self.reward_uniforms):
                raise ValueError("Maximum number of steps exceeded")
            u = self.reward_uniforms[self.current_step, action]
            reward = int(u < probabilities[action])
        else:
            reward = int(
                self.rng.binomial(n=1, p=probabilities[action])
            )

        self.current_step += 1
        return reward