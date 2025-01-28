__credits__ = ["Rushiv Arora"]

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from typing import Optional, List, Tuple


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class HalfCheetahEnvPerturbed(MujocoEnv, utils.EzPickle):
    """
    ## Description

    This environment is based on the work by P. Wawrzyński in
    ["A Cat-Like Robot Real-Time Learning to Run"](http://staff.elka.pw.edu.pl/~pwawrzyn/pub-s/0812_LSCLRR.pdf).
    The HalfCheetah is a 2-dimensional robot consisting of 9 body parts and 8
    joints connecting them (including two paws). The goal is to apply a torque
    on the joints to make the cheetah run forward (right) as fast as possible,
    with a positive reward allocated based on the distance moved forward and a
    negative reward allocated for moving backward. The torso and head of the
    cheetah are fixed, and the torque can only be applied on the other 6 joints
    over the front and back thighs (connecting to the torso), shins
    (connecting to the thighs) and feet (connecting to the shins).

    ## Action Space
    The action space is a `Box(-1, 1, (6,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                                  | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit         |
    | --- | --------------------------------------- | ----------- | ----------- | -------------------------------- | ----- | ------------ |
    | 0   | Torque applied on the back thigh rotor  | -1          | 1           | bthigh                           | hinge | torque (N m) |
    | 1   | Torque applied on the back shin rotor   | -1          | 1           | bshin                            | hinge | torque (N m) |
    | 2   | Torque applied on the back foot rotor   | -1          | 1           | bfoot                            | hinge | torque (N m) |
    | 3   | Torque applied on the front thigh rotor | -1          | 1           | fthigh                           | hinge | torque (N m) |
    | 4   | Torque applied on the front shin rotor  | -1          | 1           | fshin                            | hinge | torque (N m) |
    | 5   | Torque applied on the front foot rotor  | -1          | 1           | ffoot                            | hinge | torque (N m) |


    ## Observation Space
    Observations consist of positional values of different body parts of the
    cheetah, followed by the velocities of those individual parts (their derivatives) with all the positions ordered before all the velocities.

    By default, observations do not include the cheetah's `rootx`. It may
    be included by passing `exclude_current_positions_from_observation=False` during construction.
    In that case, the observation space will be a `Box(-Inf, Inf, (18,), float64)` where the first element
    represents the `rootx`.
    Regardless of whether `exclude_current_positions_from_observation` was set to true or false, the
    will be returned in `info` with key `"x_position"`.

    However, by default, the observation is a `Box(-Inf, Inf, (17,), float64)` where the elements correspond to the following:

    | Num | Observation                          | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | ------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | z-coordinate of the front tip        | -Inf | Inf | rootz                            | slide | position (m)             |
    | 1   | angle of the front tip               | -Inf | Inf | rooty                            | hinge | angle (rad)              |
    | 2   | angle of the second rotor            | -Inf | Inf | bthigh                           | hinge | angle (rad)              |
    | 3   | angle of the second rotor            | -Inf | Inf | bshin                            | hinge | angle (rad)              |
    | 4   | velocity of the tip along the x-axis | -Inf | Inf | bfoot                            | hinge | angle (rad)              |
    | 5   | velocity of the tip along the y-axis | -Inf | Inf | fthigh                           | hinge | angle (rad)              |
    | 6   | angular velocity of front tip        | -Inf | Inf | fshin                            | hinge | angle (rad)              |
    | 7   | angular velocity of second rotor     | -Inf | Inf | ffoot                            | hinge | angle (rad)              |
    | 8   | x-coordinate of the front tip        | -Inf | Inf | rootx                            | slide | velocity (m/s)           |
    | 9   | y-coordinate of the front tip        | -Inf | Inf | rootz                            | slide | velocity (m/s)           |
    | 10  | angle of the front tip               | -Inf | Inf | rooty                            | hinge | angular velocity (rad/s) |
    | 11  | angle of the second rotor            | -Inf | Inf | bthigh                           | hinge | angular velocity (rad/s) |
    | 12  | angle of the second rotor            | -Inf | Inf | bshin                            | hinge | angular velocity (rad/s) |
    | 13  | velocity of the tip along the x-axis | -Inf | Inf | bfoot                            | hinge | angular velocity (rad/s) |
    | 14  | velocity of the tip along the y-axis | -Inf | Inf | fthigh                           | hinge | angular velocity (rad/s) |
    | 15  | angular velocity of front tip        | -Inf | Inf | fshin                            | hinge | angular velocity (rad/s) |
    | 16  | angular velocity of second rotor     | -Inf | Inf | ffoot                            | hinge | angular velocity (rad/s) |
    | excluded |  x-coordinate of the front tip  | -Inf | Inf | rootx                            | slide | position (m)             |

    ## Rewards
    The reward consists of two parts:
    - *forward_reward*: A reward of moving forward which is measured
    as *`forward_reward_weight` * (x-coordinate before action - x-coordinate after action)/dt*. *dt* is
    the time between actions and is dependent on the frame_skip parameter
    (fixed to 5), where the frametime is 0.01 - making the
    default *dt = 5 * 0.01 = 0.05*. This reward would be positive if the cheetah
    runs forward (right).
    - *ctrl_cost*: A cost for penalising the cheetah if it takes
    actions that are too large. It is measured as *`ctrl_cost_weight` *
    sum(action<sup>2</sup>)* where *`ctrl_cost_weight`* is a parameter set for the
    control and has a default value of 0.1

    The total reward returned is ***reward*** *=* *forward_reward - ctrl_cost* and `info` will also contain the individual reward terms

    ## Starting State
    All observations start in state (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,) with a noise added to the
    initial state for stochasticity. As seen before, the first 8 values in the
    state are positional and the last 9 values are velocity. A uniform noise in
    the range of [-`reset_noise_scale`, `reset_noise_scale`] is added to the positional values while a standard
    normal noise with a mean of 0 and standard deviation of `reset_noise_scale` is added to the
    initial velocity values of all zeros.

    ## Episode End
    The episode truncates when the episode length is greater than 1000.

    ## Arguments

    No additional arguments are currently supported in v2 and lower.

    ```python
    import gymnasium as gym
    env = gym.make('HalfCheetah-v2')
    ```

    v3 and v4 take `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc.

    ```python
    import gymnasium as gym
    env = gym.make('HalfCheetah-v4', ctrl_cost_weight=0.1, ....)
    ```

    | Parameter                                    | Type      | Default              | Description                                                                                                                                                       |
    | -------------------------------------------- | --------- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | `xml_file`                                   | **str**   | `"half_cheetah.xml"` | Path to a MuJoCo model                                                                                                                                            |
    | `forward_reward_weight`                      | **float** | `1.0`                | Weight for _forward_reward_ term (see section on reward)                                                                                                          |
    | `ctrl_cost_weight`                           | **float** | `0.1`                | Weight for _ctrl_cost_ weight (see section on reward)                                                                                                             |
    | `reset_noise_scale`                          | **float** | `0.1`                | Scale of random perturbations of initial position and velocity (see section on Starting State)                                                                    |
    | `exclude_current_positions_from_observation` | **bool**  | `True`               | Whether or not to omit the x-coordinate from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies |

    ## Version History

    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3
    * v3: Support for `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc. rgb rendering comes from tracking camera (so agent does not run away from screen)
    * v2: All continuous control environments now use mujoco-py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        if exclude_current_positions_from_observation:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64
            )
        else:
            observation_space = Box(
                low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64
            )

        MujocoEnv.__init__(
            self,
            "half_cheetah.xml",
            5,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        # save nominal values
        self.gravity = self.model.opt.gravity[-1]
        
        self.bthigh_joint_stiffness = self.model.jnt_stiffness[3]
        self.bshin_joint_stiffness = self.model.jnt_stiffness[4]
        self.bfoot_joint_stiffness = self.model.jnt_stiffness[5]
        self.fthigh_joint_stiffness = self.model.jnt_stiffness[6]
        self.fshin_joint_stiffness = self.model.jnt_stiffness[7]
        self.ffoot_joint_stiffness = self.model.jnt_stiffness[8]
        
        self.actuator_ctrlrange = (-1.0, 1.0)
        self.actuator_ctrllimited = int(1)
        
        self.bthigh_joint_damping = self.model.dof_damping[3]
        self.bshin_joint_damping = self.model.dof_damping[4]
        self.bfoot_joint_damping = self.model.dof_damping[5]
        self.fthigh_joint_damping = self.model.dof_damping[6]
        self.fshin_joint_damping = self.model.dof_damping[7]
        self.ffoot_joint_damping = self.model.dof_damping[8]

        self.geom_friction = np.copy(self.model.geom_friction)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        x_position_before = self.data.qpos[0]
        
        # noise_low = -self._reset_noise_scale
        # noise_high = self._reset_noise_scale
        noise_low = -5e-3
        noise_high = 5e-3
        noise = self.np_random.uniform(low=noise_low, high=noise_high, size=action.shape)
        action_noise = action + noise
        
        self.do_simulation(action_noise, self.frame_skip)       # Stoachastic perturbed model
        # self.do_simulation(action, self.frame_skip)           # Deterministic perturbed model
        
        x_position_after = self.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        terminated = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation
    
    def test(self):
        model = self.model
        print('body_names: ', model.body_names)
        print('joint_names: ', model.joint_names)
        print('actuator_names: ', model.actuator_names)
        print('model.actuator_forcelimited', model.actuator_forcelimited)
        print('actuator_ctrlrange', model.actuator_ctrlrange)
        print('_actuator_gear', model.actuator_gear)
        print('_jnt_stiffness', model.jnt_stiffness)
        print('_dof_damping', model.dof_damping)
        print('_dof_frictionloss', model.dof_frictionloss)
        print('actuator_ctrllimited', model.actuator_ctrllimited)

    def reset(
        self,
        x_pos: float = 0.0,
        state: Optional[int] = None,  
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
        use_xml: bool = False,
        gravity: float = -9.81,
        bthigh_joint_stiffness: float = 240.0,
        bshin_joint_stiffness: float = 180.0,
        bfoot_joint_stiffness: float = 120.0,
        fthigh_joint_stiffness: float = 180.0,
        fshin_joint_stiffness: float = 120.0,
        ffoot_joint_stiffness: float = 60.0,
        front_joint_stiffness_p: float = 0.0,
        factuator_ctrlrange: Tuple[float, float] = (-1.0, 1.0),
        bactuator_ctrlrange: Tuple[float, float] = (-1.0, 1.0),
        joint_damping_p: float = 0.0,
        joint_frictionloss: float = 0.0,
        friction_p: float = 0.0,
        torso_mass: Optional[float] = None,
    ):
        ob = super().reset(seed=seed)
        # grab model
        model = self.model
        # perturb gravity in z (3rd) dimension*
        model.opt.gravity[2] = gravity
        # perturb back thigh joint*
        model.jnt_stiffness[3] = bthigh_joint_stiffness
        # perturb back shin joint*
        model.jnt_stiffness[4] = bshin_joint_stiffness
        # perturb back foot joint*
        model.jnt_stiffness[5] = bfoot_joint_stiffness
        # perturb front thigh joint*
        model.jnt_stiffness[6] = fthigh_joint_stiffness
        # perturb front shin joint*
        model.jnt_stiffness[7] = fshin_joint_stiffness
        # perturb front foot joint*
        model.jnt_stiffness[8] = ffoot_joint_stiffness
        # perturb back actuator (controller) control range*
        model.actuator_ctrllimited[0] = self.actuator_ctrllimited
        model.actuator_ctrlrange[0] = [bactuator_ctrlrange[0],
                                        bactuator_ctrlrange[1]]
        model.actuator_ctrllimited[1] = self.actuator_ctrllimited
        model.actuator_ctrlrange[1] = [bactuator_ctrlrange[0],
                                        bactuator_ctrlrange[1]]
        model.actuator_ctrllimited[2] = self.actuator_ctrllimited
        model.actuator_ctrlrange[2] = [bactuator_ctrlrange[0],
                                        bactuator_ctrlrange[1]]
        # perturb front actuator (controller) control range*
        model.actuator_ctrllimited[3] = self.actuator_ctrllimited
        model.actuator_ctrlrange[3] = [factuator_ctrlrange[0],
                                        factuator_ctrlrange[1]]
        model.actuator_ctrllimited[4] = self.actuator_ctrllimited
        model.actuator_ctrlrange[4] = [factuator_ctrlrange[0],
                                        factuator_ctrlrange[1]]
        model.actuator_ctrllimited[5] = self.actuator_ctrllimited
        model.actuator_ctrlrange[5] = [factuator_ctrlrange[0],
                                        factuator_ctrlrange[1]]
        # change joint damping in percentage
        model.dof_damping[3] = self.bthigh_joint_damping * (1 + joint_damping_p)
        model.dof_damping[4] = self.bshin_joint_damping * (1 + joint_damping_p)
        model.dof_damping[5] = self.bfoot_joint_damping * (1 + joint_damping_p)
        model.dof_damping[6] = self.fthigh_joint_damping * (1 + joint_damping_p)
        model.dof_damping[7] = self.fshin_joint_damping * (1 + joint_damping_p)
        model.dof_damping[8] = self.ffoot_joint_damping * (1 + joint_damping_p)
        # change joint frictionloss
        model.dof_frictionloss[3] = joint_frictionloss
        model.dof_frictionloss[4] = joint_frictionloss
        model.dof_frictionloss[5] = joint_frictionloss
        model.dof_frictionloss[6] = joint_frictionloss
        model.dof_frictionloss[7] = joint_frictionloss
        model.dof_frictionloss[8] = joint_frictionloss
        # friction
        model.geom_friction[:, :] = self.geom_friction * (1 + friction_p)
        # torso mass
        model.body_mass[1] = torso_mass if torso_mass is not None else model.body_mass[1]
        model.jnt_stiffness[6] = fthigh_joint_stiffness * (1 + front_joint_stiffness_p)
        model.jnt_stiffness[7] = fshin_joint_stiffness * (1 + front_joint_stiffness_p)
        model.jnt_stiffness[8] = ffoot_joint_stiffness * (1 + front_joint_stiffness_p)
        return ob
    
    def save_xml(self, savepath):
        f = open(savepath, 'w')
        self.model.save(f)
        f.close()
    
    def reset_model(self, state, x_pos):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
