from gym.envs.registration import register

register(
    id='belt-task-v0',
    entry_point='belt_task.envs:BeltTaskEnv',
)

