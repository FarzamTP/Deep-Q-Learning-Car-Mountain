import gym
from graph import Graph
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
EPISODES = 1000

STATS_EVERY = 50
SHOW_EVERY = 500

USE_EPSILON_DECAY = True
epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES // 2

model_name = f'LR: {lr} - DISCOUNT: {DISCOUNT} -' \
            f' EPISODES: {EPISODES} - Use epsilon Decay: {USE_EPSILON_DECAY} -' \
            f' EPSILON: {epsilon}'
optimal_q_table = None

epsilon_decaying_value = epsilon // (END_EPSILON_DECAYING - START_EPSILON_DECAYING)


def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / DISCRETE_os_win_size
    return tuple(discrete_state.astype(np.int))


if __name__ == '__main__':
    """
    Creates new directory in ./q_tables/
    """
    os.mkdir(f'./q_tables/{model_name}')
    os.mkdir(f'./models/{model_name}')

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
    for episode in range(1, EPISODES + 1):
        episode_reward = 0

        discrete_state = get_discrete_state(env.reset())

        done = False

        while not done:
            action = np.argmax(q_table[discrete_state])
            new_state, reward, done, _ = env.step(action)
            episode_reward += reward
            new_discrete_state = get_discrete_state(new_state)

            if episode % SHOW_EVERY == 0:
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
        if not episode % STATS_EVERY:
            average_reward = sum(ep_rewards[-STATS_EVERY:]) / STATS_EVERY
            min_reward = min(ep_rewards[-STATS_EVERY:])
            max_reward = max(ep_rewards[-STATS_EVERY:])

            aggr_ep_rewards['ep'].append(episode)
            aggr_ep_rewards['avg'].append(average_reward)
            aggr_ep_rewards['min'].append(min_reward)
            aggr_ep_rewards['max'].append(max_reward)

            print(f'Episode: {episode} avg: {average_reward} max: {max_reward} min: {min_reward}')

        """
        Saving Q-Tables for each SHOW_EVERY
        """
        if episode % STATS_EVERY == 0:
            np.save(f'./q_tables/{model_name}/{episode}-qtable.npy', q_table)
    """
    Uncomment to save the frames as gif.
    """
    # save_frames_as_gif(frames, filename="test.gif")
    env.close()

    """
    Select the q_table that produced maximum average and chose it as optimal_q_table
    """
    maximum_average = max(aggr_ep_rewards.get('avg'))
    optimal_episode_idx = aggr_ep_rewards.get('avg').index(maximum_average)
    optimal_episode = aggr_ep_rewards.get('ep')[optimal_episode_idx]
    shutil.copyfile(f'./q_tables/{model_name}/{optimal_episode}-qtable.npy',
                    f'./models/{model_name}/{optimal_episode}-qtable.npy')

    """
    Let's plot the captured episodic activities.
    """
    plt_path = f'./plots/{model_name}.png'

    plt.plot(aggr_ep_rewards.get('ep'), aggr_ep_rewards.get('avg'), label="avg rewards")
    plt.plot(aggr_ep_rewards.get('ep'), aggr_ep_rewards.get('min'), label="min rewards")
    plt.plot(aggr_ep_rewards.get('ep'), aggr_ep_rewards.get('max'), label="max rewards")
    plt.legend(loc=4)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.savefig(plt_path)
    plt.show()

    graph = Graph(q_table=q_table, save_plot_path=f'./graphs/{model_name}.png', save_plot=True)
    graph.plot()
