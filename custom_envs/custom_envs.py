import gymnasium                                            # robustness-gymnasium 0.29.2

def get_custom_env_v2(name: str, seed: int, perturb_param=None, perturb_value=None, provided_scale=None, n_envs=1) -> gymnasium.vector.VectorEnv:
    scale = provided_scale if provided_scale is not None else 5e-3
    name, version = name.split("-")
    new_name = f"{original_mj_name(name)}Perturbed-{version}"
    env = gymnasium.vector.make(new_name, num_envs=n_envs, asynchronous=False, reset_noise_scale=scale)
    # Perturbed
    if perturb_param and perturb_value:
        if isinstance(perturb_param, tuple):
            for i, single in enumerate(env.envs):
                single.reset(**{perturb_param[0]: perturb_value[0], perturb_param[1]: perturb_value[1]}, seed=seed+i)
        else:
            for i, single in enumerate(env.envs):
                single.reset(**{perturb_param: perturb_value}, seed=seed+i)
    # Unperturbed
    else:
        env.reset()
    return env


def original_mj_name(name: str) -> str:
    if name.lower() == "halfcheetah":
        return "HalfCheetah"
    else:
        return name.capitalize()