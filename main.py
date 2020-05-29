import time
import gym
import numpy as np

env = gym.make("MountainCar-v0")

lr = 0.2
DISCOUNT = 0.95
EPISODES = 10000

SHOW_EVERY = 200


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / DISCRETE_os_win_size
    return tuple(discrete_state.astype(np.int))


if __name__ == '__main__':
    """
    Initializing the Q_Table, A 20 by 20 matrix with depth of env.action_space.n (In this case 3)
    0 for going backward, 1 for staying and 2 for going forward
    """
    DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
    DISCRETE_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

    """
    Iterating over EPISODES to optimize the Q_table by using 'Bellman equation'
    as a simple value iteration update.
    """
    for episode in range(EPISODES):
        if episode % SHOW_EVERY == 0:
            print("Episode:", episode)
            render = True
        else:
            render = False

        discrete_state = get_discrete_state(env.reset())

        done = False

        while not done:
            action = np.argmax(q_table[discrete_state])
            new_state, reward, done, _ = env.step(action)
            new_discrete_state = get_discrete_state(new_state)
            if render:
                env.render()
                time.sleep(0.02)

            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,)]
                new_q = (1 - lr) * current_q + lr * (reward + DISCOUNT * max_future_q)
                q_table[discrete_state + (action,)] = new_q
            elif new_state[0] >= env.goal_position:
                q_table[discrete_state + (action,)] = 0

            discrete_state = new_discrete_state

    env.close()
