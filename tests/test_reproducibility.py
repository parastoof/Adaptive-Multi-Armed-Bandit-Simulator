import numpy as np

from agents.bandit_agents import (
    RandomAgent,
    EpsilonGreedySampleAverage,
    EpsilonGreedyConstantStep,
)
from environments.bandit_environment import BanditEnvironment


def _run_one(agent_factory, env_seed: int, agent_seed: int, n_steps: int = 50):
    env = BanditEnvironment("non_stationary", seed=env_seed)
    agent = agent_factory(seed=agent_seed)

    rewards, actions = [], []
    for _ in range(n_steps):
        a = agent.select_action()
        r = env.step(a)
        agent.update(a, r)
        rewards.append(r)
        actions.append(a)
    return np.array(rewards), np.array(actions)


def test_reproducibility_same_seed():
    factories = {
        "random":         lambda seed: RandomAgent(5, seed=seed),
        "sample_average": lambda seed: EpsilonGreedySampleAverage(5, 0.1, seed=seed),
        "constant_step":  lambda seed: EpsilonGreedyConstantStep(5, 0.1, 0.1, seed=seed),
    }

    for name, factory in factories.items():
        r1, a1 = _run_one(factory, env_seed=123, agent_seed=456)
        r2, a2 = _run_one(factory, env_seed=123, agent_seed=456)

        assert np.array_equal(r1, r2), f"{name}: rewards differ"
        assert np.array_equal(a1, a2), f"{name}: actions differ"
        print(f"PASS: {name} reproducible")


def test_different_seeds_differ():
    """Sanity check: different seeds should NOT produce identical traces."""
    f = lambda seed: EpsilonGreedySampleAverage(5, 0.1, seed=seed)
    r1, _ = _run_one(f, env_seed=1, agent_seed=1)
    r2, _ = _run_one(f, env_seed=2, agent_seed=2)
    assert not np.array_equal(r1, r2), "Different seeds produced identical traces!"
    print("PASS: different seeds produce different traces")


if __name__ == "__main__":
    test_reproducibility_same_seed()
    test_different_seeds_differ()
    print("\nAll reproducibility tests passed.")