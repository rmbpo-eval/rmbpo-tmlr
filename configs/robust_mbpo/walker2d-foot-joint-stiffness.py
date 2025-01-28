PERTURB_PARAM = "foot_joint_stiffness"
# PERTURB_VALUES = list(range(0, 50, 5))
PERTURB_VALUES = list(range(0, 400, 25))

overwrite_args = {
    "env_name": "Walker2d-v4",
    "perturb_value": PERTURB_VALUES,
    "perturb_param": [PERTURB_PARAM] * len(PERTURB_VALUES),
}