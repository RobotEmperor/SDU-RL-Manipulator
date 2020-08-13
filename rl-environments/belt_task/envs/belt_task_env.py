import gym
import numpy as np
from gym import error, spaces, utils

#ros
import rospy
from std_msgs.msg import Bool
from std_msgs.msg import Float64MultiArray
#
error_ee_pose = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
ee_velocity = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
contact_force_torque = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)
safety_violation = False

def ErrorEePoseCallBack(data):
    error_ee_pose = np.array([data.data[0], data.data[1], data.data[2], data.data[3], data.data[4], data.data[5]], dtype=np.float32)
  #rospy.loginfo(rospy.get_caller_id() + "I ErrorEePoseCallBack %f", data.data[0])

def EeVelocityCallBack(data):
    ee_velocity = np.array([data.data[0], data.data[1], data.data[2], data.data[3], data.data[4], data.data[5]], dtype=np.float32)
  #rospy.loginfo(rospy.get_caller_id() + "I EeVelocityCallBack %f", data.data[0])


def SafetyViolationCallBack(data):
  safety_violation = data.data
  #rospy.loginfo(rospy.get_caller_id() + "I SafetyViolationCallBack %d", data.data)

def  FilteredFtDataCallBack(data):
  contact_force_torque = np.array([data.data[0], data.data[1], data.data[2], data.data[3], data.data[4], data.data[5]], dtype=np.float32)


rospy.init_node('RL_node', anonymous=True)

class BeltTaskEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    #init
    self.init_ee_pose = np.array([0, 0, 0, 0, 0, 0], dtype=np.float32)

    #obs

    self.obs_state = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    obs_max_value = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 10, 10, 10, 1, 1, 1],
                             dtype=np.float32)  ##error_ee_pose, ee_velocity, force/torque 1*18

    #reward
    self.max_error_ee_pose = np.array([2, 2, 2, 2, 2, 2], dtype=np.float32)
    self.max_change_pose = np.array([2, 2, 2, 2, 2, 2], dtype=np.float32)
    self.max_contact_force_torque = np.array([10, 10, 10, 0.5, 0.5, 0.5], dtype=np.float32)
    self.reward = 0


    self.action_space = spaces.Box(
      low=-self.max_change_pose,
      high=self.max_change_pose, shape=(6,),
      dtype=np.float32
    )

    self.observation_space = spaces.Box(
      low=-obs_max_value,
      high=obs_max_value,
      dtype=np.float32
    )

    self.task_completed = False
    self.w_x = 0
    self.w_a = 0
    self.w_f = 0
    self.w_p = 0
    self.w_r = 0

    self.penalty = 0
    self.set_hyperparameters(1, 1, 0, 0, 1)
    rospy.Subscriber("/sdu/ur10e/error_ee_pose", Float64MultiArray, ErrorEePoseCallBack)
    rospy.Subscriber("/sdu/ur10e/ee_velocity", Float64MultiArray, EeVelocityCallBack)
    rospy.Subscriber("/sdu/ur10e/safety_violation", Bool, SafetyViolationCallBack)
    rospy.Subscriber("/sdu/ur10e/filtered_force_torque_data", Float64MultiArray, FilteredFtDataCallBack)

  def set_hyperparameters(self, w_x, w_a, w_f, w_p, w_r):
    self.w_x = w_x
    self.w_a = w_a
    self.w_f = w_f
    self.w_p = w_p
    self.w_r = w_r

  def norm_l_12(self, z, max_value):
    max_norm = 1

    norm = 0.5 * np.sqrt((np.sum(np.square(z)))) + np.sqrt(0.1 + np.sum(np.abs(z))**2)

    if norm >= max_norm:
      return 0
    elif norm <= 0:
      return 1
    else:
      y = -(1/(max_norm))*norm + 1
      return y


  def step(self, action):
    #rewards system
    change_pose = action

    self.modified_ee_pose = change_pose


    if np.sqrt((np.sum(np.square(error_ee_pose)))) <= 0.001:
      self.task_completed = True
      done = True
    else:
      self.task_completed = False
      done = False

    if self.task_completed:
      reward_k = 20
      done = True
    elif self.safety_violation:
      reward_k = -10
      done = True
    else:
      reward_k = 0

    self.reward = self.w_x * self.norm_l_12(error_ee_pose / self.max_error_ee_pose, self.max_error_ee_pose) + self.w_a * self.norm_l_12(change_pose / self.max_change_pose,self.max_change_pose) + self.w_f * self.norm_l_12(contact_force_torque / self.max_contact_force_torque, self.max_contact_force_torque) + self.w_p * self.penalty + self.w_r * reward_k

    #print('self.error_ee_pose :', self.error_ee_pose)

    return self._get_obs(), self.reward, False, {}

  def reset(self):
    return self._get_obs()

  def _get_obs(self):
    return self.obs_state

