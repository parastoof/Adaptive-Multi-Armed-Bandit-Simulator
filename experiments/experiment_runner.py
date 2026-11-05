import numpy as np

from agents.bandit_agents import (
    RandomAgent,
    EpsilonGreedySampleAverage,
    EpsilonGreedyConstantStep,
)
from environments.bandit_environment import BanditEnvironment


MASTER_SEED = 42
ENVIRONMENTS = ["stationary", "non_stationary"]
AGENTS = ["random", "sample_average", "constant_step"]
N_RUNS = 50
N_STEPS = 1000


def create_agent(agent_name: str, seed):
    if agent_name == "random":
        return RandomAgent(n_arms=5, seed=seed)
    if agent_name == "sample_average":
        return EpsilonGreedySampleAverage(n_arms=5, epsilon=0.1, seed=seed)
    if agent_name == "constant_step":
        return EpsilonGreedyConstantStep(
            n_arms=5, epsilon=0.1, alpha=0.1, seed=seed
        )
    raise ValueError(f"Unknown agent: {agent_name}")


def create_seed_schedule(environments, agents, n_runs, master_seed):
    root = np.random.SeedSequence(master_seed)
    env_root, agent_root = root.spawn(2)

    n_env_seeds = len(environments) * n_runs
    n_agent_seeds = len(environments) * len(agents) * n_runs

    env_sequences = env_root.spawn(n_env_seeds)
    agent_sequences = agent_root.spawn(n_agent_seeds)

    env_seeds = {}
    agent_seeds = {}

    env_idx = 0
    agent_idx = 0

    for env_name in environments:
        for run_id in range(n_runs):
            env_seeds[(env_name, run_id)] = int(
                env_sequences[env_idx].generate_state(1)[0]
            )
            env_idx += 1

            for agent_name in agents:
                agent_seeds[(env_name, agent_name, run_id)] = int(
                    agent_sequences[agent_idx].generate_state(1)[0]
                )
                agent_idx += 1

    return env_seeds, agent_seeds


def _build_reward_uniforms(env_seed: int, n_steps: int, n_arms: int) -> np.ndarray:
    """Deterministic uniform matrix shared across agents for one (env, run)."""
    rng = np.random.default_rng(env_seed)
    return rng.random((n_steps, n_arms))


def run_experiments(
    n_runs: int = N_RUNS,
    n_steps: int = N_STEPS,
    master_seed: int = MASTER_SEED,
):
    env_seeds, agent_seeds = create_seed_schedule(
        environments=ENVIRONMENTS,
        agents=AGENTS,
        n_runs=n_runs,
        master_seed=master_seed,
    )

    results = {}

    for env_name in ENVIRONMENTS:
        for agent_name in AGENTS:
            print(f"Running {agent_name} on {env_name}...")

            actions = np.zeros((n_runs, n_steps), dtype=int)
            rewards = np.zeros((n_runs, n_steps), dtype=int)
            expected_regrets = np.zeros((n_runs, n_steps), dtype=float)
            optimal_actions = np.zeros((n_runs, n_steps), dtype=int)
            cumulative_rewards = np.zeros((n_runs, n_steps), dtype=int)

            for run_idx in range(n_runs):
                env_seed = env_seeds[(env_name, run_idx)]

                reward_uniforms = _build_reward_uniforms(
                    env_seed, n_steps, n_arms=5
                )

                env = BanditEnvironment(
                    env_type=env_name,
                    seed=env_seed,
                    reward_uniforms=reward_uniforms,
                )
                env.reset()

                agent_seed = agent_seeds[(env_name, agent_name, run_idx)]
                agent = create_agent(agent_name, agent_seed)
                agent.reset()

                total_reward = 0

                for step in range(n_steps):
                    # ✅ probabilities for the CURRENT step,
                    #    read BEFORE stepping (no off-by-one)
                    probs = env.get_probabilities()
                    best_prob = float(np.max(probs))
                    # set of arms that are optimal at this step
                    optimal_mask = probs == best_prob

                    action = agent.select_action()
                    reward = env.step(action)

                    regret = best_prob - probs[action]

                    agent.update(action, reward)
                    total_reward += reward

                    actions[run_idx, step] = action
                    rewards[run_idx, step] = reward
                    expected_regrets[run_idx, step] = regret
                    # 1 if the chosen arm is *any* optimal arm
                    optimal_actions[run_idx, step] = int(
                        optimal_mask[action]
                    )
                    cumulative_rewards[run_idx, step] = total_reward

            results[(env_name, agent_name)] = {
                "actions": actions,
                "rewards": rewards,
                "expected_regrets": expected_regrets,
                "optimal_actions": optimal_actions,
                "cumulative_rewards": cumulative_rewards,
            }

    return results