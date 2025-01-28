import numpy as np


PERTURB_PARAM = "friction_p"
PERTURB_VALUES = np.arange(-0.3, 3.0, 0.3).tolist()

PERTURB_PARAM2 = "torso_mass"
PERTURB_VALUES2 = np.arange(3, 10, 0.5).tolist()

overwrite_args = {
  "env_name": "halfcheetah-v4",
  "perturb_param": [PERTURB_PARAM]*len(PERTURB_VALUES),
  "perturb_value": PERTURB_VALUES,
  "perturb_param_2": [PERTURB_PARAM2]*len(PERTURB_VALUES2),
  "perturb_value_2": PERTURB_VALUES2
}