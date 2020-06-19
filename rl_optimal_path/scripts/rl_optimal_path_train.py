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
import random
import math
import os

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Parameters
epsilon = 1  # The probability of choosing a random action (in training). This decays as iterations increase. (0 to 1)
epsilonMinimumValue = 0.001 # The minimum value we want epsilon to reach in training. (0 to 1)
nbActions = 3 # The number of actions. Since we only have left/stay/right that means 3 actions.
epoch = 10 # The number of games we want the system to run for.
hiddenSize = 100 # Number of neurons in the hidden layers.
maxMemory = 1000 # How large should the memory be (where it stores its past experiences).
batchSize = 50 # The mini-batch size for training. Samples are randomly taken from memory till mini-batch size.
gridSize = 2 # The size of the grid that the agent is going to play the game on.
nbStates = gridSize # We eventually flatten to a 1d tensor to feed the network.
discount = 0.99 # The discount is used to force the network to choose states that lead to the reward quicker (0 to 1)
learningRate = 0.0000001 # Learning Rate for Stochastic Gradient Descent (our optimizer).

"""Implements a Deep Q Network"""
X = tf.placeholder(tf.float32, [None, nbStates])
W1 = tf.Variable(tf.truncated_normal([nbStates, hiddenSize]))
b1 = tf.Variable(tf.truncated_normal([hiddenSize], stddev=0.01))
input_layer = tf.nn.relu(tf.matmul(X, W1) + b1)
W2 = tf.Variable(tf.truncated_normal([hiddenSize, hiddenSize]))
b2 = tf.Variable(tf.truncated_normal([hiddenSize], stddev=0.01))
hidden_layer = tf.nn.relu(tf.matmul(input_layer, W2) + b2)
W3 = tf.Variable(tf.truncated_normal([hiddenSize, nbActions]))
b3 = tf.Variable(tf.truncated_normal([nbActions], stddev=0.01))
output_layer = tf.matmul(hidden_layer, W3) + b3

print("initialize")
# True labels
Y = tf.placeholder(tf.float32, [None, nbActions])

# Mean squared error cost function
cost = tf.reduce_sum(tf.square(Y - output_layer)) / (2 * batchSize)

# Stochastic Gradient Decent Optimizer
optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)
#optimizer = tf.train.AdamOptimizer(1e-4).minimize(cost)

saver = tf.train.Saver()

class optimalPathEnv():
    def __init__(self):
        self.gridSize = gridSize
        self.nbStates = gridSize
        self.state = np.empty(2, dtype=np.float32)
        self.cumulative_reward = 0
        self.pre_error_state = 0

        # Returns the state of the environment.
    def observe(self):
        ob = np.zeros((self.gridSize))

        ob[0] = self.state[0] # desired
        ob[1] = self.state[1]  # robot

        ob = np.reshape(ob, (-1,self.nbStates))
        return ob

        # Resets the environment. Randomly initialise the fruit position (always at the top to begin with) and bucket.

    def reset(self):
        initial_x = random.randint(0,99)
        initial_x_robot = random.randint(0,99)
        self.state = np.array([initial_x,initial_x_robot])
        self.pre_error_state = np.abs(initial_x - initial_x_robot)
        self.cumulative_reward = 0
        return self.getState()

    def getState(self):
        stateInfo = self.state
        initial_x = stateInfo[0]
        initial_x_robot = stateInfo[1]
        return initial_x, initial_x_robot

        # Returns the award that the agent has gained for being in the current environment state.

    def getReward(self):

        reward = 0

        desired_x, robot_x = self.getState()

        error_state = desired_x - robot_x

        if error_state == 0:
            reward = 2

        if np.abs(error_state) >= self.pre_error_state:
            reward = -200
        else:
            reward = 1

        self.cumulative_reward += reward

        return self.cumulative_reward

        # else:
        #     if error_state < self.pre_error_state:
        #         reward = -1.5
        #         return reward
        #     elif error_state > self.pre_error_state:
        #         reward = 1
        #         return reward

        # if np.abs(error_state) < 20 and np.abs(error_state) >= 10:
        #     reward = -1
        #
        # if np.abs(error_state) < 10 and np.abs(error_state) >= 5:
        #     reward = 1
        #
        # if np.abs(error_state) < 5 and np.abs(error_state) > 0:
        #     reward = 2

        # if error_state == 0:
        #     reward = 1
        # else:
        #     reward = 0

    def isGameOver(self):
        desired_x, robot_x = self.getState()
        error_state = desired_x - robot_x
        check = True

        #print("error_state :: ", error_state)
        #print("pre_error_state :: ", self.pre_error_state)

        if error_state == 0:
            return True

        if self.cumulative_reward < -200:
            check = True
        else:
            check = False
        self.pre_error_state = np.abs(error_state)

        return check

    def updateState(self, action):
        if (action == 1):
            action = -1
        elif (action == 2):
            action = 1
        else:
            action = 0

        desired_x, robot_x = self.getState()
        newRobot_x = min(99, max(1,robot_x + action))  # The min/max prevents the basket from moving out of the grid.
        self.state = np.array([desired_x, newRobot_x])

        # Action can be move - direction or + direction

    def act(self, action):
        self.updateState(action)
        reward = self.getReward()
        gameOver = self.isGameOver()
        return self.observe(), reward, gameOver, self.getState()  # For purpose of the visual, I also return the state.


class ReplayMemory:
    def __init__(self, maxMemory, discount):
        self.maxMemory = maxMemory
        self.nbStates = gridSize
        self.discount = discount
        self.inputState = np.empty((self.maxMemory, self.nbStates), dtype=np.float32)
        self.actions = np.zeros(self.maxMemory, dtype=np.uint8)
        self.nextState = np.empty((self.maxMemory, self.nbStates), dtype=np.float32)
        self.gameOver = np.empty(self.maxMemory, dtype=np.bool)
        self.rewards = np.empty(self.maxMemory, dtype=np.int8)
        self.count = 0
        self.current = 0

    # Appends the experience to the memory.
    def remember(self, currentState, action, reward, nextState, gameOver):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.inputState[self.current] = currentState
        self.nextState[self.current] = nextState
        self.gameOver[self.current] = gameOver
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.maxMemory

    def getBatch(self, model, batchSize, nbActions, nbStates, sess, X):

        # We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        # batch we can (at the beginning of training we will not have enough experience to fill a batch).
        memoryLength = self.count
        chosenBatchSize = min(batchSize, memoryLength)

        inputs = np.zeros((chosenBatchSize, nbStates))
        targets = np.zeros((chosenBatchSize, nbActions))

        # Fill the inputs and targets up.
        for i in range(chosenBatchSize):
            if memoryLength == 1:
                memoryLength = 2
            # Choose a random memory experience to add to the batch.
            randomIndex = random.randrange(1, memoryLength)
            current_inputState = np.reshape(self.inputState[randomIndex], (1, self.nbStates))

            target = sess.run(model, feed_dict={X: current_inputState})

            current_nextState = np.reshape(self.nextState[randomIndex], (1, self.nbStates))
            current_outputs = sess.run(model, feed_dict={X: current_nextState})

            # Gives us Q_sa, the max q for the next state.
            nextStateMaxQ = np.amax(current_outputs)
            if (self.gameOver[randomIndex] == True):
                target[0, [self.actions[randomIndex] - 1]] = self.rewards[randomIndex]
            else:
                # reward + discount(gamma) * max_a' Q(s',a')
                # We are setting the Q-value for the action to  r + gamma*max a' Q(s', a'). The rest stay the same
                # to give an error of 0 for those outputs.
                target[0, [self.actions[randomIndex] - 1]] = self.rewards[randomIndex] + self.discount * nextStateMaxQ

            # Update the inputs and targets.
            inputs[i] = current_inputState
            targets[i] = target


        return inputs, targets

def randf(s, e):
  return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s;


if __name__ == '__main__':

    rospy.init_node('talker', anonymous=True)
    pub = rospy.Publisher('/sdu/command/task_space', Float64MultiArray, queue_size=10)
    listener = tf_data.TransformListener()
    rate = rospy.Rate(10) # 10hz

    task_space_command_msg = Float64MultiArray()
    print("start")


    #while not rospy.is_shutdown():
        #try:
            #(trans, rot) = listener.lookupTransform('/base_link', '/ee_link', rospy.Time(0))

    print("Training new model")

    # Define Environment
    env = optimalPathEnv()
    # Define Replay Memory
    memory = ReplayMemory(maxMemory, discount)
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    winCount = 0
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver.restore(sess, os.getcwd() + "/test10/model.ckpt")
        print('saved model is loaded!')


        for i in range(epoch):
            #if(rospy.is_shutdown()):
            #    break
            # Initialize the environment.
            err = 0
            env.reset()

            isGameOver = False

            # The initial state of the environment.
            currentState = env.observe()





            while ( isGameOver != True):
                if(rospy.is_shutdown()):
                    break

                action = -9999  # action initilization
                # Decides if we should choose a random action, or an action from the policy network.
                #global epsilon
                #if (randf(0, 1) <= epsilon):
                #    action = random.randrange(1,4)
                #else:
                # Forward the current state through the network.
                q = sess.run(output_layer, feed_dict={X: currentState})
                # Find the max index (the chosen action).
                #print(q)
                index =  q.argmax()
                action = index + 1
                #print(action)



                    # Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
                if (epsilon > epsilonMinimumValue):
                    epsilon = epsilon * 0.999

                #task_space_command_msg.data = [action, 0.184324, 0.5875, -3.14159, 0, 0]
                #task_space_command_msg.data = [action, 0.184324, 0, 0, 0, 0]
                #pub.publish(task_space_command_msg)
                #print("action ::  ", action, "\n\n")

                nextState, reward, gameOver, stateInfo = env.act(action)
                #print(gameOver)

                if (reward == 2):
                    winCount = winCount + 1

                memory.remember(currentState, action, reward, nextState, gameOver)

                # Update the current state and if the game is over.
                currentState = nextState
                isGameOver = gameOver




                # We get a batch of training data to train the model.
                inputs, targets = memory.getBatch(output_layer, batchSize, nbActions, nbStates, sess, X)

                #print("target!!! :: ", targets)
                #print("inputs!!! :: ", inputs)

                # Train the network which returns the error.
                _, loss = sess.run([optimizer, cost], feed_dict={X: inputs, Y: targets})

                #print("inputs!!! :: ", X)

                err = err + loss
                #print("err!!! :: ", err)
                #print("goal ::  ", stateInfo[0],"cur_robot ::  ", stateInfo[1])
                #print("action!!! :: ", action)
                #print("epsilon!!! :: ", epsilon)
                task_space_command_msg.data = [stateInfo[0], stateInfo[1], reward, 0, 0, 0]
                pub.publish(task_space_command_msg)

            print("Epoch " + str(i) + ": err = " + str(err) + ": Win count = " + str(
                winCount) + " Win ratio = " + str(
                float(winCount) / float(i + 1) * 100))
           # print("stateInfo[0] ::  ", stateInfo[0],"stateInfo[1] ::  ", stateInfo[1], "\n\n")
        # Save the variables to disk.
        save_path = saver.save(sess, os.getcwd() + "/model.ckpt")
        print("Model saved in file: %s" % save_path)

        #except (tf_data.LookupException, tf_data.ConnectivityException, tf_data.ExtrapolationException):
        #    continue

    #rate.sleep()




