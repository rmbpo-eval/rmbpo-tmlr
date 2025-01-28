import numpy as np

PERTURB_PARAM = "friction_p"
PERTURB_VALUES = [ -0.9, -0.75, -0.5, -0.4, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
#PERTURB_VALUES = np.arange(-0.9, 2, 2).tolist()

overwrite_args = {
  "env_name": "walker2d-v4",
  "perturb_param": [PERTURB_PARAM]*len(PERTURB_VALUES),
  "perturb_value": PERTURB_VALUES
}