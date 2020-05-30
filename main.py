import gym
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import time


def save_frames_as_gif(frames, path='./gifs/', filename='gym_animation.gif'):
    """
    # TODO: Remove the unnecessary save_frames_as_gif() method in deploy.
    """
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

lr = 0.1
DISCOUNT = 0.95
EPISODES = 50000

SHOW_EVERY = 200
SAVE_MODEL_EACH = 100

USE_EPSILON_DECAY = True
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

epsilon_decaying_value = epsilon // (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / DISCRETE_os_win_size
    return tuple(discrete_state.astype(np.int))


if __name__ == '__main__':
    """
    Clear the q-table directory history.
    """
    shutil.rmtree('./q_tables')
    os.mkdir('./q_tables')

    """
    Initializing the Q_Table, A 20 by 20 matrix with depth of env.action_space.n (In this case 3)
    0 for going backward, 1 for staying and 2 for going forward
    """
    DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)
    DISCRETE_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE
    q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

    ep_rewards = []
    aggr_ep_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}

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
        episode_reward = 0
        if episode % SHOW_EVERY == 0:
            render = True
        else:
            render = False

        discrete_state = get_discrete_state(env.reset())

        done = False

        while not done:
            action = np.argmax(q_table[discrete_state])
            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            new_discrete_state = get_discrete_state(new_state)
            if render:
                """
                # TODO: Uncomment below to capture frames.
                """
                # frames.append(env.render(mode="rgb_array"))
                env.render()

                """Feel free to change the sleep time if the car is too slow or too fast for you."""
                time.sleep(0.02)

            if not done:
                max_future_q = np.max(q_table[new_discrete_state])
                current_q = q_table[discrete_state + (action,)]
                new_q = (1 - lr) * current_q + lr * (reward + DISCOUNT * max_future_q)
                q_table[discrete_state + (action,)] = new_q
            elif new_state[0] >= env.goal_position:
                q_table[discrete_state + (action,)] = 0

            discrete_state = new_discrete_state

        if USE_EPSILON_DECAY:
            if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
                epsilon -= epsilon_decaying_value

        """
        Capturing the episode's activity for further tracking and model selection.
        """
        ep_rewards.append(episode_reward)
        if not episode % SHOW_EVERY:
            average_reward = sum(ep_rewards[-SHOW_EVERY:]) / len(ep_rewards[-SHOW_EVERY:])
            min_reward = min(ep_rewards[-SHOW_EVERY:])
            max_reward = max(ep_rewards[-SHOW_EVERY:])

            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['min'].append(min_reward)
            aggr_ep_rewards['max'].append(max_reward)

            print(f'Episode: {episode} avg: {average_reward} max: {max_reward} min: {min_reward}')

        """
        Saving Q-Tables for each SHOW_EVERY
        """
        if episode % SAVE_MODEL_EACH:
            np.save(f'./q_tables/{episode}-qtable.npy', q_table)
    """
    Uncomment to save the frames as gif.
    """
    # save_frames_as_gif(frames, filename="test.gif")
    env.close()

    """
    Let's plot the captured episodic activities.
    """
    plt_path = f'./plots/LR: {lr} - DISCOUNT: {DISCOUNT} -' \
               f' EPISODES: {EPISODES} - Use epsilon Decay: {USE_EPSILON_DECAY} -' \
               f' EPSILON: {epsilon}.png'
    if os.path.exists(plt_path):
        plt_path.split('.')[0] += '_NEW'

    plt.plot(aggr_ep_rewards.get('ep'), aggr_ep_rewards.get('avg'), label="avg rewards")
    plt.plot(aggr_ep_rewards.get('ep'), aggr_ep_rewards.get('min'), label="min rewards")
    plt.plot(aggr_ep_rewards.get('ep'), aggr_ep_rewards.get('max'), label="max rewards")
    plt.legend(loc=4)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(plt_path)
    plt.show()
