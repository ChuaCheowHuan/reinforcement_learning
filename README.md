# What's in this repository?

This repository contains codes that I have reproduced for various reinforcement
learning algorithms. The codes were tested on Colab.

## Implemented Algorithms

| **Algorithms**              | **Discrete**                      | **Continuous**                    | Multithreaded                     | Multiprocessing                  | **Tested on**            |
| --------------------------- | --------------------------------- | --------------------------------- |-----------------------------------|----------------------------------|--------------------------|
| DQN                         | :heavy_check_mark:                |                                   |                                   |                                  | CartPole-v0              |
| Double DQN (DDQN)           | :heavy_check_mark:                |                                   |                                   |                                  | CartPole-v0              |
| Dueling DDQN                | :heavy_check_mark:                |                                   |                                   |                                  | CartPole-v0              |
| Dueling DDQN + PER          | :heavy_check_mark:                |                                   |                                   |                                  | CartPole-v0              |
| A3C <sup>(1)</sup>          | :heavy_check_mark:                | :heavy_check_mark:                | :heavy_check_mark:                | :heavy_check_mark:<sup>(3)</sup> | CartPole-v0, Pendulum-v0 |
| DPPO <sup>(2)</sup>         |                                   | :heavy_check_mark:                |                                   | :heavy_check_mark:<sup>(3)</sup> | Pendulum-v0              |
| RND + PPO                   |                                   | :heavy_check_mark:                |                                   |                                  | MountainCarContinuous-v0 |

<sup><sup>(1): N-step returns used for critic's target.</sup></sup><br>
<sup><sup>(1): GAE used for computation of TD lambda return (for critic's target) & policy's advantage.</sup></sup><br>
<sup><sup>(3): Distributed Tensorflow & Python's multiprocessing package used.</sup></sup><br>

## Blog

Check out my [blog](https://ChuaCheowHuan.github.io/) for more information on my repositories.
