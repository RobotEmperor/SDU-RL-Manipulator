import gym
from tf2rl.algos.sac import SAC
from tf2rl.experiments.trainer import Trainer


if __name__ == '__main__':
    parser = Trainer.get_argument()
    parser = SAC.get_argument(parser)
    #parser.add_argument('--env-name', type=str, default="CartPole-v0")
    parser.add_argument('--env-name', type=str, default="belt_task:belt-task-v0")
    parser.add_argument('--model_path', type=str,
                        default='/home/yik/catkin_ws/src/SDU-RL-Manipulator/rl_motion_trj/scripts/results/',
                        help='path to save model')
    parser.add_argument('--model_name', type=str,
                        default='',
                        help='name of the saved model')

    args = parser.parse_args()

    parser.set_defaults(batch_size=100)
    parser.set_defaults(n_warmup=0)
    parser.set_defaults(max_steps=5e4)
    parser.set_defaults(episode_max_steps = 150)
    #parser.set_defaults(alpha=0.99)
    #parser.set_defaults(memory_capacity=1e3)
    parser.set_defaults(model_dir=args.model_path + args.model_name)
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
        auto_alpha=args.auto_alpha,
        lr=5e-5)


    trainer = Trainer(policy, env, args, test_env=test_env)
    trainer()
    env.ros_off()