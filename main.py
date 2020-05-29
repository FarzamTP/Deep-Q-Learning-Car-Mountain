import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()


def initialize_q_table():
    print(env.observation_space.high)
    print(env.observation_space.low)
    print(env.action_space.n)

    number_of_actions = env.action_space.n

    COL_SIZE = 20
    DISCRETE_OS_SIZE = [COL_SIZE] * len(env.observation_space.high)
    print(DISCRETE_OS_SIZE)
    DISCRETE_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
    print(DISCRETE_os_win_size)
    q_table = np.random.uniform(low=-2, high=0, size=(COL_SIZE, COL_SIZE, number_of_actions))
    print(q_table.shape)
    print(q_table)


if __name__ == '__main__':
    initialize_q_table()

    done = False

    while not done:
        action = 2
        new_state, reward, done, _ = env.step(action)
        print("new_state:", new_state)
        print("reward:", reward)
        print("done:", done)
        env.render()

    env.close()