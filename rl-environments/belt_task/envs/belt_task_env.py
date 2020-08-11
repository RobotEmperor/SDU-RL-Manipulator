import gym
import numpy as np
from gym import error, spaces, utils


class BeltTaskEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    self.init_ee_pose = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    self.desired_ee_pose = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

    self.max_error_ee_pose = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)
    self.max_change_pose = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3], dtype=np.float32)

    self.max_ee_velocity = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    self.contact_force_torque = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
    self.max_contact_force_torque = np.array([5, 5, 5, 0.5, 0.5, 0.5], dtype=np.float32)
    self.ee_pose = self.init_ee_pose
    self.reward = 0


    self.action_space = spaces.Box(
      low=-self.max_change_pose,
      high=self.max_change_pose, shape=(6,),
      dtype=np.float32
    )

    high = self.max_error_ee_pose ##error_pose!!

    self.observation_space = spaces.Box(
      low=-high,
      high=high,
      dtype=np.float32
    )

    self.task_completed = False
    self.safety_violation = False
    self.w_x = 0
    self.w_a = 0
    self.w_f = 0
    self.w_p = 0
    self.w_r = 0

    self.penalty = 0
    self.pre_defined_reward = 0

    self.pre_ee_pose = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

    self.set_hyperparameters(1, 1, 0, 0, 0)

    self.error_ee_pose = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

  def set_hyperparameters(self, w_x, w_a, w_f, w_p, w_r):
    self.w_x = w_x
    self.w_a = w_a
    self.w_f = w_f
    self.w_p = w_p
    self.w_r = w_r

  def norm_l_12(self, z, max_value):
    max_norm = 0.5 * np.sqrt(6) + np.sqrt(0.1 + 6 ** 2)

    norm = 0.5 * np.sqrt((np.sum(np.square(z)))) + np.sqrt(0.1 + np.sum(np.abs(z))**2)

    #print('norm :',norm)
    #print('max_norm :', max_norm)

    if norm >= max_norm:
      return -0.1
    elif norm <= 0:
      return 1
    else:
      y = -(1/(max_norm))*norm + 1
      #y = np.clip(norm, -1, 0)
      return y


  def step(self, action):
    #rewards system
    change_pose = action

    self.ee_pose = self.init_ee_pose + change_pose

    self.error_ee_pose = self.desired_ee_pose - self.ee_pose

    ee_velocity = (self.pre_ee_pose - self.ee_pose)/0.002

    check_vel_max = np.greater_equal(ee_velocity, self.max_ee_velocity)

    for i in check_vel_max:
      if i:
        self.safety_violation = True
      else:
        self.safety_violation = False

    if np.sum(np.abs(self.error_ee_pose)) >= 0 and np.sum(np.abs(self.error_ee_pose)) <= 0.001:
      self.task_completed = True
      done = True
    else:
      done = False

    if self.task_completed:
      reward_k = 200
    elif self.safety_violation:
      reward_k = -10
    else:
      reward_k = 0

    self.reward = self.w_x * self.norm_l_12(self.error_ee_pose / self.max_error_ee_pose, self.max_error_ee_pose) + self.w_a * self.norm_l_12(change_pose / self.max_change_pose,self.max_change_pose) # + self.w_f * self.norm_l_12(self.contact_force_torque / self.max_contact_force_torque, self.max_contact_force_torque) + self.w_p * 0.1 + self.w_r * reward_k

    self.pre_ee_pose = self.ee_pose

    #print('self.ee_pose :', self.ee_pose)
    #print('self.action :', action)

    #print('self.action :', action)
    #print('self.reward :', self.reward)
    #if self.reward > 0.95:
    #print('self.error_norm :', np.sqrt(np.sum(np.square(self.error_ee_pose))))

    return self._get_obs(), self.reward, done, {}

  def reset(self):
    #self.desired_ee_pose = np.random.rand(6)
    #self.desired_ee_pose = np.clip(self.desired_ee_pose, -self.max_error_ee_pose/2, self.max_error_ee_pose/2)
    self.init_ee_pose = np.random.rand(6)
    self.init_ee_pose = np.clip(self.init_ee_pose, -self.max_error_ee_pose/2, self.max_error_ee_pose/2)
    self.error_ee_pose = self.desired_ee_pose - self.init_ee_pose
    return self._get_obs()

  def _get_obs(self):
    return self.error_ee_pose