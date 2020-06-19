import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math
import os

from rl_optimal_path_train import optimalPathEnv, gridSize, X, W1, b1, input_layer, W2, b2, hidden_layer, W3, b3, output_layer, Y, cost, optimizer
maxGames = 100
env = optimalPathEnv()
winCount = 0
loseCount = 0
numberOfGames = 0


# Add ops to save and restore all the variables.
saver = tf.train.Saver()


with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, os.getcwd() + "/test10/model.ckpt")
    print('saved model is loaded!')

    while (numberOfGames < maxGames):
        numberOfGames = numberOfGames + 1

        # The initial state of the environment.
        isGameOver = False
        desired_x, cur_x = env.reset()
        currentState = env.observe()
        #winCount = 0
        #loseCount = 0

        while (isGameOver != True):
            # Forward the current state through the network.
            q = sess.run(output_layer, feed_dict={X: currentState})
            # Find the max index (the chosen action).
            index = q.argmax()
            action = index + 1
            nextState, reward, gameOver, stateInfo = env.act(action)
            desired_x = stateInfo[0]
            cur_x = stateInfo[1]

            # Count game results
            if (desired_x == cur_x):
                winCount = winCount + 1
            if (gameOver==True and reward < -200):
                loseCount = loseCount + 1

            print("numberOfGames :: ", numberOfGames,winCount,loseCount)
            print("desired_x ::", desired_x)
            print("cur_x ::", cur_x)
            currentState = nextState
            isGameOver = gameOver
        #print("isGameOver ::", isGameOver)
    print("winCount ::",winCount)
    print("loseCount ::",loseCount)


