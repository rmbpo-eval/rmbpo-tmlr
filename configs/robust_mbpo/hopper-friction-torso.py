import numpy as np

overwrite_args = {
    "env_name": "Hopper-v4",
    "perturb_param_2": ["torso_mass"] * len(np.arange(2.5, 5.0, 0.25).tolist()),
    "perturb_param": ["friction_p"] * 9,
    "perturb_value_2": np.arange(2.5, 5.0, 0.25).tolist(),
    "perturb_value": [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]
}