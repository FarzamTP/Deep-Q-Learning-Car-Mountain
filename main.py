import gym
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import time


def save_frames_as_gif(frames, path='./gifs/', filename='gym_animation.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 100.0, frames[0].shape[0] / 100.0), dpi=50)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    print("File saved!")
    return


env = gym.make("MountainCar-v0")

lr = 0.2
DISCOUNT = 0.95
EPISODES = 5000

SHOW_EVERY = 2000

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decaying_value = epsilon // (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


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
    Frames list captures the frames of the successful car episode,
    and the save_frames_as_gif() will convert it to gif.
    """
    frames = []

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
                frames.append(env.render(mode="rgb_array"))
                env.render()

                """Feel free to change the sleep time if the car is too slow or too fast for you."""
                time.sleep(0.02)

            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,)]
                new_q = (1 - lr) * current_q + lr * (reward + DISCOUNT * max_future_q)
                q_table[discrete_state + (action,)] = new_q
            elif new_state[0] >= env.goal_position:
                print(f"We made it on episode {episode}")
                q_table[discrete_state + (action,)] = 0

            discrete_state = new_discrete_state

        if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
            epsilon -= epsilon_decaying_value

    save_frames_as_gif(frames, filename="phase_one_epsilon_decay.gif")
    env.close()
