from __future__ import division, print_function, absolute_import

import os
import numpy as np
import torch

from gymnasium import utils, spaces
from gymnasium.envs.mujoco import MujocoEnv


class SwimmerEnv(MujocoEnv, utils.EzPickle):
    MODEL_IN, MODEL_OUT = 12, 10  # you can adjust depending on your dynamics model input/output
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 20,
    }

    def __init__(self):
        utils.EzPickle.__init__(self)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(dir_path, "assets", "swimmer.xml")

        MujocoEnv.__init__(self, xml_path,  frame_skip=4, observation_space=None, render_mode=None)


        obs_dim = self.model.nq + self.model.nv
        act_dim = self.model.nu

        high_obs = np.inf * np.ones(obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)

        high_act = np.ones(act_dim, dtype=np.float32)
        self.action_space = spaces.Box(-high_act, high_act, dtype=np.float32)

    def step(self, action):
        xpos_before = self.data.qpos[0].copy()
        self.do_simulation(action, self.frame_skip)
        xpos_after = self.data.qpos[0].copy()
        ob = self._get_obs()

        forward_reward = (xpos_after - xpos_before) / self.dt
        reward = forward_reward

        terminated = False
        truncated = False
        info = dict(
            reward_forward=forward_reward,
            x_position=xpos_after,
            x_velocity=(xpos_after - xpos_before) / self.dt,
        )
        return ob, reward, terminated or truncated, info

    def reset_model(self):
        qpos = self.init_qpos + np.random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        qvel = self.init_qvel + np.random.randn(self.model.nv) * 0.1
        self.set_state(qpos, qvel)
        return self._get_obs()
    
    def seed(self, s):
        self.seed = s
        np.random.seed(s)
        
    def _get_obs(self):
        return np.concatenate([self.data.qpos.flat, self.data.qvel.flat])

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = v.model.stat.extent

    # --- cost functions for MBRL ---
    @staticmethod
    def obs_cost_fn_cost(obs):
        # encourage high forward velocity
        if isinstance(obs, np.ndarray):
            return -obs[:, 2]  # assuming qvel[0] is forward velocity
        else:
            return -obs[:, 2]

    @staticmethod
    def ac_cost_fn_cost(acs):
        if isinstance(acs, np.ndarray):
            return 0.0001 * np.sum(np.square(acs), axis=1)
        else:
            return 0.0001 * torch.sum(torch.square(acs), dim=1)

    @staticmethod
    def obs_preproc(obs):
        return obs

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs
