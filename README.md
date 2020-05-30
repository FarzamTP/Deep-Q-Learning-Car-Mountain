# Q-Learning

### Phase one
This is the first phase of the project that focuses on training the car to reach the peak by initializing and updating the Q-Table.
Here we use the [Bellman equation](https://en.wikipedia.org/wiki/Bellman_equation) as a simple [value iteration update](https://en.wikipedia.org/wiki/Markov_decision_process#Value_iteration).

![Bellman equation](./images/Bellman-Equation.svg)

![initial q_table status](./graphs/LR:%200.1%20-%20DISCOUNT:%200.95%20-%20EPISODES:%201000%20-%20Use%20epsilon%20Decay:%20True%20-%20EPSILON:%200.5.png)

* As you see in the image, you see that the distribution of Q table values are so vide that cannot be used in order to make a decision. But by taking a brief look at the below graph, shows that the final Q-table values are in a particular distribution that can be used to make a decision.

![final q table status](./graphs/LR:%200.1%20-%20DISCOUNT:%200.95%20-%20EPISODES:%2060000%20-%20Use%20epsilon%20Decay:%20True%20-%20EPSILON:%200.5.png)

#### Result (Phase one)
##### Not Using Epsilon Decay
![phase one gif](./gifs/phase_one.gif)

As it's obvious, the car at episode 1, has no idea what to do. But after only 500 episodes 
it understands that by making progress to the right, he'll gain a point!
But there is an interesting point there:
***Although the car receives its reward by reaching the peak, as it's shown in the gif, it tried to minimize the spent time. To do that, at first it decreases amount of the path in goes forward and uses its gained velocity to reach the top!***

##### Using Epsilon Decay
![phase one gif](./gifs/phase_one_epsilon_decay.gif)

Although the car reaches the peak in a quite acceptable time, by using epsilon decay we make model to ***explore*** more in order to find a better approach!
And as it's shown in above gif, the car minimizes it's spent time to reach the peak.

![LR: 0.1 - DISCOUNT: 0.95 - EPISODES: 1000 - Use epsilon Decay: True - EPSILON: 0.5](./plots/LR:%200.05%20-%20DISCOUNT:%200.95%20-%20EPISODES:%201000%20-%20Use%20epsilon%20Decay:%20True%20-%20EPSILON:%200.7.png)

![LR: 0.1 - DISCOUNT: 0.95 - EPISODES: 2000 - Use epsilon Decay: True - EPSILON: 0.5](./plots/LR:%200.1%20-%20DISCOUNT:%200.95%20-%20EPISODES:%201000%20-%20Use%20epsilon%20Decay:%20True%20-%20EPSILON:%200.5.png)

![LR: 0.1 - DISCOUNT: 0.95 - EPISODES: 4000 - Use epsilon Decay: True - EPSILON: 0.5](./plots/LR:%200.1%20-%20DISCOUNT:%200.95%20-%20EPISODES:%204000%20-%20Use%20epsilon%20Decay:%20True%20-%20EPSILON:%200.5.png)

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
