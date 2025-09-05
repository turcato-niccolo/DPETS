from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy as np
from gymnasium.envs.mujoco import MujocoEnv
from gym import utils, spaces
import torch


class CartDoublePendulumStabEnv(MujocoEnv, utils.EzPickle):
    PENDULUM_LENGTH = [0.6, 0.6]   # length of each link
    MODEL_IN, MODEL_OUT = 9, 6     # positions + velocities
    OBS_ADD_DIM = 2                # because we’ll encode 2 angles (sin/cos)

    def __init__(self):
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        MujocoEnv.__init__(self, '%s/assets/cart_double_pendulum.xml' % dir_path, frame_skip=2, observation_space=None)
        
        obs_dim = self.model.nq + self.model.nv
        act_dim = self.model.nu

        high_obs = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high_obs, high_obs)

        high_act = np.ones(act_dim, dtype=np.float32) * 3
        self.action_space = spaces.Box(-high_act, high_act)

        # Initial configuration: both links pointing up
        self.init_qpos = np.array([0.0, 0.0, 0.0])  # [cart pos, theta1, theta2]

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        # reward: keep end-effector close to (0, sum of lengths)
        target = np.array([0.0, sum(CartDoublePendulumStabEnv.PENDULUM_LENGTH)])
        x, _, y = self.data.site_xpos[0]

        reward = -np.linalg.norm(np.array([x,y]) - target)

        done = False
        return ob, reward, done, {"reward": reward}

    def reset_model(self):
        qpos = self.init_qpos + np.random.normal(0, 0.1, np.shape(self.init_qpos))
        qvel = self.init_qvel + np.random.normal(0, 0.1, np.shape(self.init_qvel))
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([self.unwrapped.data.qpos, self.unwrapped.data.qvel]).ravel()

    def seed(self, s):
        self.seed = s
        np.random.seed(s)

    @staticmethod
    def _get_ee_pos(x):
        """
        Compute end-effector pos from [cart, theta1, theta2, ...]
        """
        x0, theta1, theta2 = x[0], x[1], x[2]
        l1, l2 = CartDoublePendulumStabEnv.PENDULUM_LENGTH
        x_ee = x0 - l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)
        y_ee = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
        return np.array([x_ee, y_ee])

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = v.model.stat.extent

    @staticmethod
    def obs_cost_fn_cost(obs):
        target = np.array([0.0, sum(CartDoublePendulumStabEnv.PENDULUM_LENGTH)])
        if isinstance(obs, np.ndarray):
            return -np.exp(-np.sum(
                np.square(CartDoublePendulumStabEnv._get_ee_pos_cost(obs) - target), axis=1
            ) / (sum(CartDoublePendulumStabEnv.PENDULUM_LENGTH) ** 2))
        else:
            target_t = torch.tensor(target, device=obs.device)
            return -torch.exp(-torch.sum(
                torch.square(CartDoublePendulumStabEnv._get_ee_pos_cost(obs) - target_t), dim=1
            ) / (sum(CartDoublePendulumStabEnv.PENDULUM_LENGTH) ** 2))

    @staticmethod
    def ac_cost_fn_cost(acs):
        if isinstance(acs, np.ndarray):
            return 0.01 * np.sum(np.square(acs), axis=1)
        else:
            return 0.01 * torch.sum(torch.square(acs), dim=1)

    @staticmethod
    def _get_ee_pos_cost(obs):
        """
        Batched version of end-effector computation.
        """
        l1, l2 = CartDoublePendulumStabEnv.PENDULUM_LENGTH
        x0, theta1, theta2 = obs[:, :1], obs[:, 1:2], obs[:, 2:3]
        if isinstance(obs, np.ndarray):
            return np.concatenate([
                x0 - l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2),
                -l1 * np.cos(theta1) - l2 * np.cos(theta1 + theta2)
            ], axis=1)
        else:
            return torch.cat([
                x0 - l1 * torch.sin(theta1) - l2 * torch.sin(theta1 + theta2),
                -l1 * torch.cos(theta1) - l2 * torch.cos(theta1 + theta2)
            ], dim=1)

    @staticmethod
    def obs_preproc(obs):
        """
        Replace [theta1, theta2] with [sin(theta1), cos(theta1), sin(theta2), cos(theta2)]
        """
        if isinstance(obs, np.ndarray):
            if obs.ndim == 1:
                return np.concatenate([
                    np.sin(obs[1:2]), np.cos(obs[1:2]),
                    np.sin(obs[2:3]), np.cos(obs[2:3]),
                    obs[:1], obs[3:]
                ], axis=-1)
            return np.concatenate([
                np.sin(obs[:, 1:2]), np.cos(obs[:, 1:2]),
                np.sin(obs[:, 2:3]), np.cos(obs[:, 2:3]),
                obs[:, :1], obs[:, 3:]
            ], axis=1)
        else:
            if obs.ndim == 3:
                return torch.cat([
                    torch.sin(obs[:, :, 1:2]), torch.cos(obs[:, :, 1:2]),
                    torch.sin(obs[:, :, 2:3]), torch.cos(obs[:, :, 2:3]),
                    obs[:, :, :1], obs[:, :, 3:]
                ], dim=2)
            return torch.cat([
                torch.sin(obs[:, 1:2]), torch.cos(obs[:, 1:2]),
                torch.sin(obs[:, 2:3]), torch.cos(obs[:, 2:3]),
                obs[:, :1], obs[:, 3:]
            ], dim=1)

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs
