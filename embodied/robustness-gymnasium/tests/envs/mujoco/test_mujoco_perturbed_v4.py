import pytest
import gymnasium
import copy
import numpy as np

import gymnasium.envs.mujoco
import gymnasium.wrappers.time_limit

# List of MuJoCo environments that end with '-s'
mujoco_envs = [
    'HopperPerturbed-v4',
    'Walker2dPerturbed-v4',
    'HalfCheetahPerturbed-v4'
]

@pytest.mark.parametrize("env_id", mujoco_envs)
def test_mujoco_env_reset(env_id):
    env: gymnasium.wrappers.time_limit.TimeLimit = gymnasium.make(env_id) # type: ignore
    try:
        env.reset()
        assert True
    except Exception as e:
        pytest.fail(f"Environment {env_id} failed to reset without disturbances: {e}")
    finally:
        env.close()

@pytest.mark.parametrize("env_id", mujoco_envs)
def test_seeding_consistency(env_id):
    env_1: gymnasium.envs.mujoco.MujocoEnv = gymnasium.make(env_id) # type: ignore
    env_2: gymnasium.envs.mujoco.MujocoEnv = gymnasium.make(env_id) # type: ignore
    seed = 42
    env_1.reset(seed=seed)
    env_2.reset(seed=seed)
    env_1.action_space.seed(seed)
    env_2.action_space.seed(seed)
    try:
        assert np.all(env_1.step(env_1.action_space.sample())[0] == env_2.step(env_2.action_space.sample())[0])
    except Exception as e:
        pytest.fail(f"Environment {env_id} failed to maintain consistency after seeding: {e}")
    finally:
        env_1.close()
        env_2.close()

@pytest.mark.parametrize("env_id", mujoco_envs)
def test_reset_perturbed_modify_gravity(env_id):
    env: gymnasium.envs.mujoco.MujocoEnv = gymnasium.make(env_id)   # type: ignore
    env_unwrapped: gymnasium.envs.mujoco.MujocoEnv = env.unwrapped  # type: ignore
    testvalue = -11.72
    try:
        gravity_before = env_unwrapped.model.opt.gravity[2]
        env.reset(gravity=np.float64(testvalue))
        assert env_unwrapped.model.opt.gravity[2] == np.float64(testvalue) 
        assert env_unwrapped.model.opt.gravity[2] != gravity_before    
    except Exception as e:
        pytest.fail(f"Environment {env_id} failed to reset with gravity disturbance: {e}")
    finally:
        env.close()

@pytest.mark.parametrize("env_id", mujoco_envs)
def test_reset_perturbed_modify_friction(env_id):
    env_unwrapped: gymnasium.envs.mujoco.MujocoEnv = gymnasium.make(env_id).unwrapped # type: ignore
    testvalue = 0.5
    try:
        friction_before = np.copy(env_unwrapped.model.geom_friction)
        env_unwrapped.reset(friction_p = testvalue) 
        assert np.isclose(env_unwrapped.model.geom_friction, friction_before * (1 + testvalue)).all()
        assert np.any(env_unwrapped.model.geom_friction != friction_before)
    except Exception as e:
        pytest.fail(f"Environment {env_id} failed to reset with friction disturbance: {e}")
    finally:
        env_unwrapped.close()

@pytest.mark.parametrize("env_id", mujoco_envs)
def test_reset_not_perturbed(env_id):
    env_1: gymnasium.envs.mujoco.MujocoEnv = gymnasium.make(env_id).unwrapped # type: ignore
    env_2: gymnasium.envs.mujoco.MujocoEnv = gymnasium.make(env_id).unwrapped # type: ignore
    env_1.reset()
    try:
        difference = False
        parameters = ['opt', 'geom_friction', 'jnt_stiffness', 'qpos_spring', 'actuator_ctrllimited', 'actuator_ctrlrange', 'dof_damping', 'dof_frictionloss', 'body_mass']
        for parameter in parameters:
            if isinstance(getattr(env_1.model, parameter), np.ndarray):
                difference = difference or np.any(getattr(env_1.model, parameter) != getattr(env_2.model, parameter))
            else:
                difference = difference or getattr(env_1.model, parameter) != getattr(env_2.model, parameter)
        assert not difference
    except Exception as e:
        pytest.fail(f"Environment {env_id} failed to remain unaltered after normal reset: {e}")
    finally:
        env_1.close()
        env_2.close()