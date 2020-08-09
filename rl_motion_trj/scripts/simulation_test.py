import numpy as np
import gym
from tf2rl.algos.sac import SAC
from tf2rl.experiments.trainer import Trainer
import tensorflow as tf

parser = Trainer.get_argument()
parser = SAC.get_argument(parser)
# parser.add_argument('--env-name', type=str, default="CartPole-v0")
parser.add_argument('--env-name', type=str, default="Pendulum-v0")
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--render', type=bool, default=False,
                    help='set gym environment to render display')
parser.add_argument('--verbose', type=bool, default=False,
                    help='log execution details')
parser.add_argument('--model_path', type=str, default='/home/yik/catkin_ws/src/SDU-RL-Manipulator/rl_motion_trj/scripts/results/',
                    help='path to save model')
parser.add_argument('--model_name', type=str,
                    default=f'',
                    help='name of the saved model')

parser.set_defaults(batch_size=100)
parser.set_defaults(n_warmup=10000)
parser.set_defaults(max_steps=3e5)

args = parser.parse_args()

env = gym.make(args.env_name)
test_env = gym.make(args.env_name)

policy = SAC(
    state_shape=env.observation_space.shape,
    action_dim=env.action_space.high.size,
    gpu=-1,
    memory_capacity=args.memory_capacity,
    max_action=env.action_space.high,
    batch_size=args.batch_size,
    n_warmup=args.n_warmup,
    alpha=args.alpha,
    auto_alpha=args.auto_alpha)

to_restore = policy

policy_checkpointer = tf.train.Checkpoint(policy=to_restore)
policy_checkpointer.restore(tf.train.latest_checkpoint(args.model_path + args.model_name)).expect_partial()

#print(actor_net.trainable_weights[0])


#fake_policy = tf.train.Checkpoint(bias=to_restore)
#fake_policy = tf.train.Checkpoint(bias=to_restore)
#status = to_restore.restore(tf.train.latest_checkpoint(args.model_path + args.model_name)).expect_partial()

while True:

    # Instantiate the environment.
    # TODO: fix this when env.action_space is not `Box`

    # Observe state
    current_state = env.reset()

    episode_reward = 0
    done = False
    while not done:

        action_ = to_restore.get_action(current_state, True)
        action = action_

        # Execute action, observe next state and reward
        next_state, reward, done, _ = env.step(action)

        episode_reward +=  reward

        # Update current state
        current_state = next_state
        #env.render()

    print(episode_reward)
