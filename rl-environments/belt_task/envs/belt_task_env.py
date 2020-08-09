import gym
import numpy as np
from gym import error, spaces, utils


class BeltTaskEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.init_ee_pose = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)
    self.desired_ee_pose = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
#    self.current_ee_pose = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
#    self.error_ee_pose = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    self.max_ee_pose = np.array([2, 2, 2, 2, 2, 2], dtype=np.float32)
#    self.change_pose = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=np.float32)
    self.max_change_pose = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
#    self.contact_force_torque = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
#    self.max_contact_force_torque = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    #self.reward = np.array([1], dtype=np.float32)
    self.reward = 0

    self.action_space = spaces.Box(
      low=-self.max_change_pose,
      high=self.max_change_pose, shape=(6,),
      dtype=np.float32
    )

    high = np.array([1, 1, 1, 1, 1, 1], dtype=np.float32)

    self.observation_space = spaces.Box(
      low=-high,
      high=high,
      dtype=np.float32
    )

    self.safety_violation = False
    self.w_x = 0
    self.w_a = 0
    self.w_f = 0
    self.w_p = 0
    self.w_r = 0

    self.penalty = 0
    self.pre_defined_reward = 0

    self.set_hyperparameters(1, 0, 0, 0, 0)

  def set_hyperparameters(self, w_x, w_a, w_f, w_p, w_r):
    self.w_x = w_x
    self.w_a = w_a
    self.w_f = w_f
    self.w_p = w_p
    self.w_r = w_r

  def norm_l_12(self, z):
    max_norm = np.linalg.norm(self.max_ee_pose)
    norm = 0.5*(np.linalg.norm(z)**2) + np.sqrt(0.1 + z**2)
    norm = np.linalg.norm(norm)

    if norm >= max_norm:
      return 1
    elif norm <= 0:
      return 0
    else:
      y = (1/(max_norm))*norm
      return y

  def step(self, action):
    ee_pose = self.state  # th := theta
    #rewards system

    error_ee_pose = self.desired_ee_pose - ee_pose
    ##
    change_pose = np.clip(action, -self.max_change_pose, self.max_change_pose)

    self.reward = self.w_x * self.norm_l_12(error_ee_pose / self.max_ee_pose)

    self.state = change_pose

    return self._get_obs(), self.reward, False, {}

  def reset(self):
    self.state = self.init_ee_pose
    return self._get_obs()

  def _get_obs(self):
    #x, y, z, eaa_x, eaa_y, eaa_z = self.state
    return  self.state

"""
    self.reward[num] = self.w_x * self.norm_l_12(
      self.error_ee_pose[num] / self.max_ee_pose[num]) + self.w_a * self.norm_l_12(
      self.change_pose[num] / self.max_change_pose[num]) + self.w_f * self.norm_l_12(
      self.contact_force_torque[num] / self.max_contact_force_torque[
        num]) + self.w_p * self.penalty + self.w_r * self.pre_defined_reward
"""