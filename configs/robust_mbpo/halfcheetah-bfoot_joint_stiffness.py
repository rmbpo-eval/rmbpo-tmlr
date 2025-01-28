import numpy as np


PERTURB_PARAM = "bfoot_joint_stiffness"
PERTURB_VALUES = np.arange(0.0, 121, 10).tolist()

overwrite_args = {
  "env_name": "halfcheetah-v4",
  "perturb_param": [PERTURB_PARAM]*len(PERTURB_VALUES),
  "perturb_value": PERTURB_VALUES,
}