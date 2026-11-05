from pathlib import Path

import matplotlib.pyplot as plt

from storage.database import get_metric_curves


BASE_DIR = Path(__file__).resolve().parents[1]
PLOT_DIR = BASE_DIR / "plots"

ENVIRONMENTS = ["stationary", "non_stationary"]
AGENTS = ["random", "sample_average", "constant_step"]

AGENT_STYLE = {
    "random":          {"linestyle": ":",  "color": "#7f7f7f"},
    "sample_average":  {"linestyle": "-",  "color": "#1f77b4"},
    "constant_step":   {"linestyle": "--", "color": "#d62728"},
}

METRICS = {
    "average_reward_per_step": {
        "title": "Average Reward per Step",
        "ylabel": "Average Reward",
        "filename": "average_reward_per_step.png",
    },
    "average_cumulative_reward": {
        "title": "Average Cumulative Reward",
        "ylabel": "Cumulative Reward",
        "filename": "average_cumulative_reward.png",
    },
    "optimal_action_rate": {
        "title": "Optimal Action Rate",
        "ylabel": "Optimal Action Rate",
        "filename": "optimal_action_rate.png",
    },
    "average_cumulative_expected_regret": {
        "title": "Average Cumulative Expected Regret",
        "ylabel": "Cumulative Expected Regret",
        "filename": "average_cumulative_expected_regret.png",
    },
}


def _plot_metric(df, metric_name, config, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=False)

    for ax, env in zip(axes, ENVIRONMENTS):
        env_data = df[df["environment"] == env]

        for agent in AGENTS:
            agent_data = env_data[env_data["agent"] == agent]
            if agent_data.empty:
                continue

            style = AGENT_STYLE[agent]
            ax.plot(
                agent_data["step"],
                agent_data[metric_name],
                label=agent,
                linewidth=1.8,
                **style,
            )

        if env == "non_stationary":
            ax.axvline(
                x=500.5,
                linestyle="--",
                color="black",
                alpha=0.7,
                label="Probability change",
            )

        ax.set_title(env.replace("_", " ").title())
        ax.set_xlabel("Step")
        ax.set_ylabel(config["ylabel"])
        ax.grid(alpha=0.3)
        ax.legend(loc="best")

    fig.suptitle(config["title"], fontsize=15)
    fig.tight_layout()

    out_path = output_dir / config["filename"]
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


def generate_plots(experiment_id: str):
    df = get_metric_curves(experiment_id)

    if df.empty:
        raise RuntimeError(
            f"No data found for experiment: {experiment_id}"
        )

    out_dir = PLOT_DIR / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRetrieved {len(df):,} metric rows from SQLite.")

    for name, cfg in METRICS.items():
        _plot_metric(df, name, cfg, out_dir)

    print(f"\nAll plots saved under: {out_dir}")