import numpy as np

PERTURB_PARAM = "friction_p"
PERTURB_VALUES = [-1.0, -0.75, -0.5, -0.25, 0, 0.2, 0.25, 0.5, 0.75, 1.0]

PERTURB_PARAM2 = "torso_mass"
PERTURB_VALUES2 = np.arange(1.0, 9, 1).tolist()

overwrite_args = {
  "env_name": "walker2d-v4",
  "perturb_param": [PERTURB_PARAM]*len(PERTURB_VALUES),
  "perturb_value": PERTURB_VALUES,
  "perturb_param_2": [PERTURB_PARAM2]*len(PERTURB_VALUES2),
  "perturb_value_2": PERTURB_VALUES2
}