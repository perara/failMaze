import gym
import gym_maze
from sklearn.preprocessing import minmax_scale
import pandas
import numpy as np
import json
from example.logger import logger
from example.dqn.dqn_example_4 import DQN


def preprocess(state):
    new_state = state
    #new_state = np.reshape(state, state.shape[:2])
    #new_state = minmax_scale(new_state)
    #new_state = np.reshape(new_state, (1, ) + new_state.shape + (1, ))
    new_state = np.reshape(new_state, (1, ) + new_state.shape)
    return new_state


if __name__ == '__main__':

    env_list = [
        "maze-arr-6x6-full-deterministic-v0",
    ]

    env = gym.make(env_list[-1])
    env.reset()

    timeout_steps = 1000

    epochs = 500

    batch_size = 32
    train_epochs = 10
    memory_size = 100000

    agent = DQN(
        env.observation_space,
        env.action_space,
        memory_size=memory_size,
        batch_size=batch_size,
        train_epochs=train_epochs,
        e_min=0,
        e_max=1.0,
        e_steps=100000,
        lr=1e-2,
        discount=0.99
    )
    agent.model.summary()
    try:
        agent.load("./model_weights.h5")
    except:
        pass

    while True:
        for env_name in env_list:
            print("Creating env %s" % env_name)
            env = gym.make(env_name)
            env.reset()
            agent.epsilon = agent.epsilon_max

            epoch = 0
            while epoch < epochs:
                epoch += 1

                # Reset environment
                state = env.reset()
                state = preprocess(state)
                terminal = False
                timestep = 0

                while not terminal:
                    timestep += 1


                    # Draw action from distribution
                    action = agent.act(state)



                    # Perform action in environment
                    next_state, reward, terminal, info = env.step(action)
                    next_state = preprocess(next_state)
                    if reward > 0:
                        reward = 100
                    else:
                        reward = -0.01

                    # Memorize
                    agent.remember(state, action, reward, next_state, terminal)

                    state = next_state

                    if timestep > timeout_steps:
                        terminal = True

                    if terminal:
                        pass

                if len(agent.memory) > agent.batch_size:
                    agent.replay(q_table=env.env.q_table)
                env.render()
                logger.info(json.dumps({
                    "epoch": epoch,
                    "steps": timestep,
                    "optimal": info["optimal_path"],
                    "epsilon": agent.epsilon,
                    "loss": agent.average_loss(),
                    "terminal": terminal,
                    "replay": len(agent.memory),
                    "env": env_name
                }))

