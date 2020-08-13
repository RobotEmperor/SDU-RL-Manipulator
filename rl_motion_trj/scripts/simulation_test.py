import numpy as np
import gym
from tf2rl.algos.sac import SAC
from tf2rl.experiments.trainer import Trainer
import tensorflow as tf

parser = Trainer.get_argument()
parser = SAC.get_argument(parser)
# parser.add_argument('--env-name', type=str, default="CartPole-v0")
parser.add_argument('--env-name', type=str, default="belt_task:belt-task-v0")
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

args = parser.parse_args()

parser.set_defaults(batch_size=100)
parser.set_defaults(n_warmup=0)
parser.set_defaults(max_steps=5e4)
parser.set_defaults(episode_max_steps=150)
parser.set_defaults(model_dir = args.model_path + args.model_name)

args = parser.parse_args()

env = gym.make(args.env_name)
test_env = gym.make(args.env_name)

policy_ = SAC(
    state_shape=env.observation_space.shape,
    action_dim=env.action_space.high.size,
    gpu=-1,
    memory_capacity=args.memory_capacity,
    max_action=env.action_space.high,
    batch_size=args.batch_size,
    n_warmup=args.n_warmup,
    alpha=args.alpha,
    auto_alpha=args.auto_alpha,
    lr=1e-5)

trainer = Trainer(policy_, env, args, test_env=test_env)

#trainer()

current_steps = 0
max_steps = 5
total_steps = 0
episode_max_steps = 10

while total_steps <= max_steps:
    current_steps = 0
    episode_reward = 0
    # Instantiate the environment.

    # Observe state
    current_state = env.reset()
    #env.desired_ee_pose = np.random.rand(6)
    #done = False
    total_steps += 1
    while current_steps <= episode_max_steps:

        current_steps += 1
        action_ = trainer._policy.get_action(current_state, True)

        # Execute action, observe next state and reward
        next_state, reward, done, _ = env.step(action_)

        episode_reward += reward

        # Update current state
        current_state = next_state
        print('step:', current_steps, 'current_state:', current_state)

    #print(current_state)

    print(episode_reward)
