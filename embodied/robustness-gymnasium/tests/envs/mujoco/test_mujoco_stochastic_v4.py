import pytest
import gymnasium
import copy
import numpy as np

import gymnasium.envs.mujoco

# List of MuJoCo environments that end with '-s'
mujoco_envs = [
    'HopperResample-v4',
    'Walker2dResample-v4',
    'HalfCheetahResample-v4',
]

@pytest.mark.parametrize("env_id", mujoco_envs)
def test_mujoco_env_reset(env_id):
    env = gymnasium.make(env_id)
    try:
        env.reset()
        assert True
    except Exception as e:
        pytest.fail(f"Environment {env_id} failed to reset: {e}")
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
def test_reset_ds_identical(env_id):
    seed = 42
    env: gymnasium.wrappers.time_limit.TimeLimit = gymnasium.make(env_id) # type: ignore
    env.action_space.seed(seed=seed)
    action = env.action_space.sample()
    action_noiseless = np.concatenate([action, np.zeros(action.shape)])
    try:
        # Normal reset
        s, _ = env.reset(seed=seed)
        s_org, x_pos = copy.deepcopy(s), np.array([env.unwrapped.data.qpos[0]])
        
        # Reset to specific state + perform noiseless action
        env.reset(state=s_org, x_pos=x_pos)
        observation_1, reward_1, _, _, _ = env.step(action_noiseless)

        # Repeat previous step
        env.reset(state=s_org, x_pos=x_pos)
        observation_2, reward_2, _, _, _ = env.step(action_noiseless)

        # Should be identical
        assert np.all(observation_1 == observation_2)
        assert reward_1 == reward_2

    except Exception as e:
        pytest.fail(f"Environment {env_id} failed to reset to a fixed state: {e}")
    finally:
        env.close()

@pytest.mark.parametrize("env_id", mujoco_envs)
def test_reset_ds_noisy(env_id):
    seed = 42
    env: gymnasium.wrappers.time_limit.TimeLimit = gymnasium.make(env_id) # type: ignore
    env.action_space.seed(seed=seed)
    action = env.action_space.sample()
    try:
        # Normal reset
        s, _ = env.reset(seed=seed)
        s_org, x_pos = copy.deepcopy(s), np.array([env.unwrapped.data.qpos[0]])
        
        # Reset to specific state + perform action
        env.reset(state=s_org, x_pos=x_pos)
        observation_1, reward_1, _, _, _ = env.step(action)

        # Repeat previous step
        env.reset(state=s_org, x_pos=x_pos)
        observation_2, reward_2, _, _, _ = env.step(action)

        # Should not be identical, due to stochastic mdp
        assert np.any(observation_1 != observation_2)
        assert reward_1 != reward_2

    except Exception as e:
        pytest.fail(f"Environment {env_id} failed to be stochastic: {e}")
    finally:
        env.close()

@pytest.mark.parametrize("env_id", mujoco_envs)
def test_reset_ds_resample(env_id):
    seed = 42
    env: gymnasium.wrappers.time_limit.TimeLimit = gymnasium.make(env_id) # type: ignore
    env.action_space.seed(seed=seed)
    action = env.action_space.sample().astype(np.float64) # For identical concatenation to noise vector
    try:
        # Normal reset
        s, _ = env.reset(seed=seed)
        s_org, x_pos = copy.deepcopy(s), np.array([env.unwrapped.data.qpos[0]])
        
        # Reset to specific state + perform action
        env.reset(state=s_org, x_pos=x_pos)
        observation_1, reward_1, _, _, info = env.step(action)

        # Reset to original state and perform action with same noise vector
        env.reset(state=s_org, x_pos=x_pos)
        observation_2, reward_2, _, _, _ = env.step(np.concat([action, info['noise']]))

        # Should be identical
        assert np.all(observation_1 == observation_2)
        assert reward_1 == reward_2        

    except Exception as e:
        pytest.fail(f"Environment {env_id} failed to be deterministic with a fixed noise vector: {e}")
    finally:
        env.close()