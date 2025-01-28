import gymnasium.envs
import gymnasium.vector
import numpy as np
import torch
import gymnasium

def fnIQM(data: np.ndarray, axis=0):
    """
    Calculate the Interquartile Mean (IQM) of the given data.
    Ignore 25 and 75 percentile outliers and calculate the mean of the remaining data.
    """
    q1_values = np.expand_dims(np.percentile(data, 25, axis=axis), -1)
    q3_values = np.expand_dims(np.percentile(data, 75, axis=axis), -1)
    local_data = data.copy()
    mask = (local_data >= q1_values) & (local_data <= q3_values)
    local_data[~mask] = np.nan
    result = np.nanmean(local_data, axis=axis)
    return result


@torch.no_grad()
def evaluate_v2(eval_envs: gymnasium.vector.VectorEnv, perturb_params, perturb_values, agent, descriptions, max_trajectory_length):
    """
    Evaluates the performance of an agent across one or more vectorized Gymnasium environments with optional environment perturbations.
    Args:
        eval_envs (gymnasium.vector.VectorEnv or List[gymnasium.vector.VectorEnv]):
            One or more vectorized environments to evaluate. Non-list inputs will be wrapped in a list.
        perturb_params (List[Union[str, Tuple[str, str]]]):
            Perturbation parameter(s) to pass to the environment reset method. Each element can be a string or a tuple of strings.
        perturb_values (List[Union[float, Tuple[float, float]]]):
            Corresponding perturbation value(s) to pass to the environment reset method. Each element can be a float or a tuple of floats.
        agent:
            An object representing the agent, which must have a 'select_action' method returning a dictionary containing 'action'.
        descriptions (str or List[str]):
            One or more descriptive labels corresponding to each environment in 'eval_envs'. Non-list inputs will be wrapped in a list.
        max_trajectory_length (int):
            The maximum number of steps to run each environment before truncation.
    Returns:
        dict:
            A dictionary containing aggregated performance metrics for each evaluated environment. 
            Keys include mean, median, variance, standard deviation, worst, best, and average trajectory length for each item in 'descriptions', 
            as well as an overall mean return.
    """
    eval_envs = eval_envs if isinstance(eval_envs, list) else [eval_envs]
    descriptions = descriptions if isinstance(descriptions, list) else [descriptions]
    
    ret_info = {}
    for i, vec_env in enumerate(eval_envs):
        num_envs = vec_env.num_envs
        traj_returns = np.zeros(num_envs, dtype=np.float32)
        traj_lengths = np.zeros(num_envs, dtype=np.int32)

        obs_info = []
        for env in vec_env.envs:
            if isinstance(perturb_params[0], tuple):
                # Two perturbations
                obs_info.append(env.unwrapped.reset(**{perturb_params[i][0]: perturb_values[i][0], perturb_params[i][1]: perturb_values[i][1]}))
            else:
                # Single perturbation
                obs_info.append(env.reset(**{perturb_params[i]: perturb_values[i]}))
        obs = np.array([info[0] for info in obs_info], dtype=np.float32)

        dones = np.zeros(num_envs, dtype=bool)
        for _ in range(max_trajectory_length):
            actions = agent.select_action(obs, deterministic=True)["action"]
            next_obs, rewards, termination, truncation, _ = vec_env.step(actions)
            dones = np.logical_or.reduce((dones, termination, truncation))
            rewards *= (1 - dones)
            traj_returns += rewards
            traj_lengths += (1 - dones)
            obs = next_obs
            if np.all(dones):
                break

        var = np.var(traj_returns, ddof=1) if num_envs > 1 else 0
        std = np.sqrt(var)
        d = descriptions[i]
        mean_return = np.mean(traj_returns)
        ret_info.update({
            f"performance/eval_return_{d}": mean_return,
            f"performance/eval_median_{d}": np.median(traj_returns),
            f"performance/eval_variance_{d}": var,
            f"performance/eval_std_{d}": std,
            f"performance/eval_worst_{d}": np.amin(traj_returns),
            f"performance/eval_best_{d}": np.amax(traj_returns),
            f"performance/eval_length_{d}": np.mean(traj_lengths),
        })

    ret_info["performance/eval_return"] = np.mean([ret_info[f"performance/eval_return_{d}"] for d in descriptions])
    return ret_info


def to_grid(array: np.ndarray, dim: int):
    return array.reshape(dim, dim)

LIMITS = {
    "hopper-v4": (400, 4000),
    "walker2d-v4": (400, 5000),
    "halfcheetah-v4": (1000, 12000),
}