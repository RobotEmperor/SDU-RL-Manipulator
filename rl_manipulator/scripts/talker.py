#!/usr/bin/env python


import rospy
import tf as tf_data
from std_msgs.msg import String
from std_msgs.msg import Float64MultiArray
import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

import tensorflow as tf


from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 32  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

class optimalPathEnv(py_environment.PyEnvironment):
  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(6,), dtype=np.float32, minimum=[0,0,0,0,0,0], maximum=[1,1,1,1,1,1], name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(6,), dtype=np.float32, name='observation', minimum=[0,0,0,0,0,0], maximum = [1,1,1,1,1,1])
    self._state = np.array([0,0,0,0,0,0])
    self._episode_ended = False
    self._cur_state = np.array([0,0,0,0,0,0])
    self._desired_state = np.array([0, 0, 0, 0, 0, 0])

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = np.array([0,0,0,0,0,0])
    self._episode_ended = False
    return ts.restart(np.array(self._state, dtype=np.float32))


  def _get_state(self, current_observation):
      self._cur_state = current_observation

  def _desired_state(self, desired_observation):
      self._desired_state = desired_observation

  def _step(self, action):

    if self._episode_ended:
      # The last action ended the episode. Ignore the current action and start
      # a new episode.
      return self.reset()

    # TODO ACTION (random)
    action = np.random.rand(6)

    # publish action to simulation
    error_state = self._desired_state - self._cur_state

    if np.abs(np.mean(error_state)) > 0.1:
        reward = 0
        self._episode_ended = True

    if np.abs(np.mean(error_state)) < 0.1 and np.abs(np.mean(error_state)) >0.05:
        reward = 0.5
        return ts.termination(np.array(cur_state, dtype=np.float32), reward)

    if np.abs(np.mean(error_state)) < 0.05:
        reward = 1
        return ts.termination(np.array(self._cur_state, dtype=np.float32), reward)
    else:
        ts.transition(
          np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)


if __name__ == '__main__':

    rospy.init_node('talker', anonymous=True)
    pub = rospy.Publisher('/sdu/command/task_space', Float64MultiArray, queue_size=10)
    listener = tf_data.TransformListener()
    rate = rospy.Rate(10) # 10hz

    task_space_command_msg = Float64MultiArray()

    while not rospy.is_shutdown():
        try:
            (trans, rot) = listener.lookupTransform('/base_link', '/ee_link', rospy.Time(0))
            print(trans)
        except (tf_data.LookupException, tf_data.ConnectivityException, tf_data.ExtrapolationException):
            continue

        task_space_command_msg.data = [0, 0, 0, 0, 0, 0]
        pub.publish(task_space_command_msg)

        rate.sleep()



