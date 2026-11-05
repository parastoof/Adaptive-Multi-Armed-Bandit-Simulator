import numpy as np


def calculate_metrics(results: dict) -> dict:
    """
    Compute per-step metric curves directly from in-memory simulation
    results. Used to cross-check the SQL-derived curves.
    """
    metrics = {}

    for key, data in results.items():
        rewards = data["rewards"]
        cumulative_rewards = data["cumulative_rewards"]
        optimal_actions = data["optimal_actions"]
        expected_regrets = data["expected_regrets"]

        cumulative_regrets = np.cumsum(expected_regrets, axis=1)

        metrics[key] = {
            "average_reward_per_step": rewards.mean(axis=0),
            "average_cumulative_reward": cumulative_rewards.mean(axis=0),
            "optimal_action_rate": optimal_actions.mean(axis=0),
            "average_cumulative_expected_regret":
                cumulative_regrets.mean(axis=0),
        }

    return metrics