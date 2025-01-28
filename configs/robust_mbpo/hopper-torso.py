import numpy as np

overwrite_args = {
    "env_name": "Hopper-v4",
    "perturb_param": ["torso_mass"] * len(np.arange(2.5, 6.5, 0.5).tolist()),
    "perturb_value": np.arange(2.5, 6.5, 0.5).tolist(),
}
