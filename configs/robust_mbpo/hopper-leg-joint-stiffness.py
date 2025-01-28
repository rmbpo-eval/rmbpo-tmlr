import numpy as np

overwrite_args = {
    "env_name": "Hopper-v4",
    "perturb_value": list(np.arange(0.0, 800.0, 50.0)),
    "perturb_param": ["leg_joint_stiffness"] * len(np.arange(0.0, 800.0, 50.0)),
    # "perturb_value": list(np.arange(0.0, 1000.0, 100.0)),
    # "perturb_param": ["leg_joint_stiffness"] * len(np.arange(0.0, 1000.0, 100.0)),
}