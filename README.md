# Q-Learning

## Problem
`MountainCar-v0` is a `Gym` environment that provides us a simple environment that can take action as input and by using `step()` method, returns a new state, reward and whether the goal is reached or not.
Here is an screenshot of the initial environment.
<p align='center'>
  <img src='./images/initial_state.png' alt='initial_state_of_car'>
  <p align='center'>Initial State of The Environment</p>
</p>

In this project, aim is to implement a Q-Learning algorithm in the first phase, and also develope a deep Q-Learning algorithm using `Keras`.

### Phase one
This is the first phase of the project that focuses on training the car to reach the peak by updating the Q-Table.
Here we use the [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation) as a simple [value iteration update](https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration).

![Bellman equation](./images/Bellman-Equation.svg)

<p align='center'>
  <img src='./graphs/LR:%200.1%20-%20DISCOUNT:%200.95%20-%20EPISODES:%201000%20-%20Use%20epsilon%20Decay:%20True%20-%20EPSILON:%200.5.png' alt='Initial Q-Table'>
</p>

As it can be understood from the above scatter graph, the green dots represent the action taken by the agent in the previous action. In the plot, the agent chooses a action which is almost random.
But after some iterations through episodes, the agent learns that taking specific actions is specific locations, leads it to a receive a reward!
When the agent understands the solution, It keeps exploiting it. Ofcourse it can lead the model to gain reward constantly, But the model doesn't know that it can gain more reward by taking different actions, which is known as exploration.
This the one of the foundamental problems of reinforcement learning, known as ***Exploration Explotation Dilemma***.


<p align='center'>
  <img src='./graphs/LR:%200.1%20-%20DISCOUNT:%200.95%20-%20EPISODES:%2060000%20-%20Use%20epsilon%20Decay:%20True%20-%20EPSILON:%200.5.png' alt='Final Q-Table'>
</p>

#### Result (Phase one)
##### Not Using Epsilon Decay
<p align='center'>
  ![phase one gif](./gifs/phase_one.gif)
</p>

As it's obvious, the car at episode 1, has no idea what to do. But after only 500 episodes 
it understands that by making progress to the right, he'll gain a point!
But there is an interesting point there:
***Although the car receives its reward by reaching the peak, as it's shown in the gif, it tried to minimize the spent time. To do that, at first it decreases amount of the path in goes forward and uses its gained velocity to reach the top!***

##### Using Epsilon Decay

<p align='center'>
  ![phase one gif](./gifs/phase_one_epsilon_decay.gif)
</p>


Although the car reaches the peak in a quite acceptable time, by using epsilon decay we make model to ***explore*** more in order to find a better approach!
And as it's shown in above gif, the car minimizes it's spent time to reach the peak.


![LR: 0.1 - DISCOUNT: 0.95 - EPISODES: 1000 - Use epsilon Decay: True - EPSILON: 0.5](./plots/LR:%200.05%20-%20DISCOUNT:%200.95%20-%20EPISODES:%201000%20-%20Use%20epsilon%20Decay:%20True%20-%20EPSILON:%200.7.png)


![LR: 0.1 - DISCOUNT: 0.95 - EPISODES: 2000 - Use epsilon Decay: True - EPSILON: 0.5](./plots/LR:%200.1%20-%20DISCOUNT:%200.95%20-%20EPISODES:%201000%20-%20Use%20epsilon%20Decay:%20True%20-%20EPSILON:%200.5.png)


![LR: 0.1 - DISCOUNT: 0.95 - EPISODES: 5000 - Use epsilon Decay: True - EPSILON: 0.5](./plots/LR:%200.1%20-%20DISCOUNT:%200.95%20-%20EPISODES:%205000%20-%20Use%20epsilon%20Decay:%20True%20-%20EPSILON:%200.5.png)


![LR: 0.1 - DISCOUNT: 0.95 - EPISODES: 10000 - Use epsilon Decay: True - EPSILON: 0.5](./plots/LR:%200.1%20-%20DISCOUNT:%200.95%20-%20EPISODES:%2010000%20-%20Use%20epsilon%20Decay:%20True%20-%20EPSILON:%200.5.png)


![LR: 0.1 - DISCOUNT: 0.95 - EPISODES: 15000 - Use epsilon Decay: True - EPSILON: 1](./plots/LR:%200.1%20-%20DISCOUNT:%200.95%20-%20EPISODES:%2015000%20-%20Use%20epsilon%20Decay:%20True%20-%20EPSILON:%201.png)


![LR: 0.1 - DISCOUNT: 0.95 - EPISODES: 40000 - Use epsilon Decay: True - EPSILON: 0.5](./plots/LR:%200.1%20-%20DISCOUNT:%200.95%20-%20EPISODES:%2040000%20-%20Use%20epsilon%20Decay:%20True%20-%20EPSILON:%200.5.png)


![LR: 0.1 - DISCOUNT: 0.95 - EPISODES: 50000 - Use epsilon Decay: True - EPSILON: 0.5](./plots/LR:%200.1%20-%20DISCOUNT:%200.95%20-%20EPISODES:%2050000%20-%20Use%20epsilon%20Decay:%20True%20-%20EPSILON:%200.5.png)


#### Final Result
* This is the final result gained by training the model for 60000 EPISODES:

![Final gif](./gifs/LR:%200.1%20-%20DISCOUNT:%200.95%20-%20EPISODES:%2060000%20-%20Use%20epsilon%20Decay:%20True%20-%20EPSILON:%200.5.gif)

### How to use:
First clone the repository:
```shell script
$ git clone https://github.com/FarzamTP/Deep-Q-Learning-Car-Mountain.git
$ cd Deep-Q-Learning-Car-Mountain
```
To setup the `virtual environment` and `activating` it:
```shell script
$ python3 -m venv venv
$ source venv/bin/activate
```

And to install the requirements:
```shell script
(venv)$ pip3 install -r requirements.txt
```

and run the `main.py` script:
```shell script
(venv)$ python3 main.py
```
