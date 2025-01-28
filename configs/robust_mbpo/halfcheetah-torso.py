import numpy as np

PERTURB_PARAM = "torso_mass"
PERTURB_VALUES = np.arange(1, 19, 1).tolist()

overwrite_args = {
  "env_name": "HalfCheetah-v4",
  "perturb_param": [PERTURB_PARAM]*len(PERTURB_VALUES),
  "perturb_value": PERTURB_VALUES
}