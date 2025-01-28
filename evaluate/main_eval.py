# Standard imports
import os
import glob
import time
import click
import gym.spaces
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from scipy.stats import bootstrap

# Unstable Baselines imports
import unstable_baselines.common.util as util
from unstable_baselines.common.logger import Logger
from unstable_baselines.model_based_rl.mbpo.agent import MBPOAgent as Agent
from unstable_baselines.common.util import (
    set_device_and_logger,
    load_config,
    set_global_seed,
)

# Custom imports
from util.util import evaluate_v2, fnIQM, LIMITS
from custom_envs.custom_envs import get_custom_env_v2
import gym

@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.argument("config-path", type=str, required=True)
@click.option("--log-dir", default=os.path.join("logs", "sac"), help="Output Log directory")
@click.option("--gpu", type=int, default=-1, help="GPU ID to use (e.g. --gpu 0)")
@click.option("--seed", type=int, default=30, help="The random seed, used for the evaluation environments")
@click.option("--info", type=str, default="")
@click.option("--load-dir", type=str, default="", help="Directory that contains one or multiple policy weights to evaluate together")
@click.option("--num-eval-runs", type=int, default=10, help="The amount of evaluation runs per perturbation parameter, per policy")
@click.option("--metric", type=str, default="mean", help="Metric to aggregate results over trained policies, and calculate confidence intervals")
@click.option("--no-plot", is_flag=True, help="Disable plotting")
@click.option("--no-save", is_flag=True, help="Disable saving the results")
@click.argument("args", nargs=-1)
def main(
    config_path, log_dir, gpu, seed, info, load_dir, num_eval_runs, metric, no_plot, no_save, args
):
    before = time.time()

    # Load configuration
    args = load_config(config_path, args)

    # Get last directory from load_dir
    agent_type = os.path.basename(os.path.normpath(load_dir))
    environment_name = args['env_name']
    disturbance_type = args['perturb_param'][0] + ('_' + args['perturb_param_2'][0] if args.get('perturb_param_2') else '')

    # Set global seed
    set_global_seed(seed)

    # Get environment name and perturbation parameters
    env_name = args["env_name"]
    perturb_params = args["perturb_param"]
    perturb_values = args["perturb_value"]

    if args.get("perturb_param_2"):
        perturb_params_2 = args["perturb_param_2"]
        perturb_values_2 = args["perturb_value_2"]
        # All combinations
        perturb_params = [
            (i, j) for i in perturb_params for j in perturb_params_2
        ]
        perturb_values = [
            (i, j) for i in perturb_values for j in perturb_values_2
        ]
        print(
            "\033[94m INFO: \033[0m You are using combined robustness parameters. This may take a long time to evaluate!"
        )

    # Initialize logger
    logger = Logger(
        log_dir, env_name, seed=seed, info_str=info, print_to_terminal=False
    )
    logger.log_str(f"Logging to {logger.log_path}")

    # Set device and logger
    set_device_and_logger(gpu, logger)

    # Log parameters
    logger.log_str_object("parameters", log_dict=args)

    # Initialize environments
    logger.log_str("Initializing Environments")
    eval_envs = []
    eval_env_descriptions = []
    for param, value in zip(perturb_params, perturb_values):
        eval_envs.append(
            get_custom_env_v2(
                env_name,
                seed=seed,
                perturb_param=param,
                perturb_value=value,
                n_envs=num_eval_runs,
            )
        )
        eval_env_descriptions.append(f"{param}_{value}")

    # Get observation and action spaces
    obs_space = eval_envs[0].single_observation_space
    action_space = eval_envs[0].single_action_space

    # Initialize agent
    logger.log_str("Initializing Agent")

    # Load all the .pt files in the directory
    policy_files = glob.glob(os.path.join(load_dir, "*.pt"))

    # Initialize list to store results
    eval_mean_returns = []

    # Evaluate each policy file
    for i, policy_file in enumerate(policy_files):
        # Print loading progress
        percent = (i / len(policy_files)) * 100
        num_hashes = int(percent // 2)
        load_bar = "#" * num_hashes + "-" * (50 - num_hashes)
        print(
            f"Loading: [{load_bar}] {percent:.2f}% of weights complete"
        )

        # Initialize agent and load weights
        old_gym_obs_space = gym.spaces.Box(float("-inf"), float("inf"), obs_space.shape, dtype=np.float32)
        old_gym_action_space = gym.spaces.Box(-1.0, 1.0, shape=action_space.shape, dtype=np.float32)
        agent = Agent(
            old_gym_obs_space, old_gym_action_space, env_name=env_name, **args["agent"]
        )
        if not os.path.exists(policy_file):
            print(
                f"\033[31mLoad path not found: {policy_file}\033[0m"
            )
            exit(0)
        agent.load_state_dict(
            torch.load(policy_file, map_location=util.device)
        )

        # Evaluate agent
        h = args["trainer"]["max_trajectory_length"]
        eval_results = evaluate_v2(
            eval_envs,
            perturb_params,
            perturb_values,
            agent,
            eval_env_descriptions,
            h,
        )

        # Append evaluation results
        eval_mean_returns.append(
            [
                eval_results[f"performance/eval_return_{desc}"]
                for desc in eval_env_descriptions
            ]
        )

    # Calculate metric
    eval_mean_returns = np.array(eval_mean_returns)
    if metric == "iqm":
        bootstrap_fn = fnIQM
        means = fnIQM(np.expand_dims(np.transpose(eval_mean_returns), 1), axis=-1)
    elif metric == "mean":
        bootstrap_fn = np.mean
        means = np.mean(eval_mean_returns, axis=0)
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Bootstrap confidence intervals
    bootstrap_result = bootstrap(
        (eval_mean_returns,),
        bootstrap_fn,
        n_resamples=10000,
        axis=0,
        confidence_level=0.95,
    )
    confidence_low = bootstrap_result.confidence_interval.low
    confidence_high = bootstrap_result.confidence_interval.high

    if not no_save:
        if not os.path.exists("results_data"):
            os.makedirs("results_data")
        if not os.path.exists(f"results_data/{environment_name}"):
            os.makedirs(f"results_data/{environment_name}")
        if not os.path.exists(f"results_data/{environment_name}/{agent_type}"):
            os.makedirs(f"results_data/{environment_name}/{agent_type}")
        # Save the final data
        file_name = f"{disturbance_type}-{metric}.pkl"
        file_name = os.path.join(
            "results_data", environment_name, agent_type, file_name
        )
        with open(file_name, "wb") as f:
            pickle.dump(
                (
                    perturb_params,
                    perturb_values,
                    eval_mean_returns,
                    confidence_low,
                    confidence_high,
                ),
                f,
            )

    if not no_plot:
        # Plotting results
        if isinstance(perturb_params[0], tuple):
            # 2D case
            x_vals = np.array([pv[0] for pv in perturb_values])
            y_vals = np.array([pv[1] for pv in perturb_values])

            x_unique = np.unique(x_vals)
            y_unique = np.unique(y_vals)

            X, Y = np.meshgrid(x_unique, y_unique, indexing="ij")
            Z = np.zeros_like(X, dtype=float)

            for i in range(len(means)):
                idx_x = np.where(x_unique == x_vals[i])[0][0]
                idx_y = np.where(y_unique == y_vals[i])[0][0]
                Z[idx_x, idx_y] = means[i]

            # Plot heatmap
            plt.figure()
            plt.pcolormesh(
                X, Y, Z, shading="auto", cmap="viridis"
            )
            plt.colorbar(label="Mean Return")
            plt.xlabel("Perturbation Parameter 1")
            plt.ylabel("Perturbation Parameter 2")
            plt.clim(LIMITS[env_name.lower()])
            plt.title("Mean Return Heatmap")
        else:
            # 1D case
            plt.figure()
            plt.plot(
                perturb_values,
                means,
                marker="o",
                color="orange",
            )
            plt.fill_between(
                perturb_values,
                confidence_low,
                confidence_high,
                color="orange",
                alpha=0.1,
            )
            plt.ylim(bottom=0)
            plt.ylabel("Evaluation Mean Return")
            plt.xlabel("Parameter Value")
            plt.title(
                "Evaluation Mean Return vs. Distortion Parameter Value"
            )
            plt.legend()
            plt.grid(True)

        after = time.time()
        print(f"Time taken: {after - before:.2f} seconds")
        plt.show()


if __name__ == "__main__":
    main()