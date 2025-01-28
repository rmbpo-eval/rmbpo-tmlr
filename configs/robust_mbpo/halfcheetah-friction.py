import numpy as np


PERTURB_PARAM = "friction_p"
PERTURB_VALUES = np.arange(-0.9, 2.5, 0.2).tolist()

overwrite_args = {
  "env_name": "halfcheetah-v4",
  "perturb_param": [PERTURB_PARAM]*len(PERTURB_VALUES),
  "perturb_value": PERTURB_VALUES,
}