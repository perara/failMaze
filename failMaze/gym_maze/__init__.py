from gym.envs.registration import register
import gym_maze.envs.maze_env

for env in dir(gym_maze.envs.maze_env):

    if "Maze" not in env or "Env" not in env:
        continue

    clazz_env = getattr(gym_maze.envs.maze_env, env)

    register(
        id=clazz_env.id,
        entry_point='gym_maze.envs:%s' % env
    )
