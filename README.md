# What's in this repository?

This repository contains codes that I have reproduced (while learning RL) for
various reinforcement learning algorithms. The codes were tested on Colab.

If Github is not loading the Jupyter notebooks, a known Github issue,
click [here](https://nbviewer.jupyter.org/github/ChuaCheowHuan/reinforcement_learning/tree/master/)
to view the notebooks on Jupyter's nbviewer.

---

## Implemented Algorithms

| **Algorithms**              | **Discrete**                      | **Continuous**                    | Multithreaded                     | Multiprocessing                  | **Tested on**            |
| --------------------------- | --------------------------------- | --------------------------------- |-----------------------------------|----------------------------------|--------------------------|
| DQN                         | :heavy_check_mark:                |                                   |                                   |                                  | CartPole-v0              |
| Double DQN (DDQN)           | :heavy_check_mark:                |                                   |                                   |                                  | CartPole-v0              |
| Dueling DDQN                | :heavy_check_mark:                |                                   |                                   |                                  | CartPole-v0              |
| Dueling DDQN + PER          | :heavy_check_mark:                |                                   |                                   |                                  | CartPole-v0              |
| A3C <sup>(1)</sup>          | :heavy_check_mark:                | :heavy_check_mark:                | :heavy_check_mark:                | :heavy_check_mark:<sup>(3)</sup> | CartPole-v0, Pendulum-v0 |
| DPPO <sup>(2)</sup>         |                                   | :heavy_check_mark:                |                                   | :heavy_check_mark:<sup>(3)</sup> | Pendulum-v0              |
| RND + PPO                   |                                   | :heavy_check_mark:                |                                   |                                  | MountainCarContinuous-v0 <sup>(4)</sup>, Pendulum-v0 <sup>(5)</sup> |

<sup><sup>(1): N-step returns used for critic's target.</sup></sup><br>
<sup><sup>(1): GAE used for computation of TD lambda return (for critic's target) & policy's advantage.</sup></sup><br>
<sup><sup>(3): Distributed Tensorflow & Python's multiprocessing package used.</sup></sup><br>
<sup><sup>(4): State featurization (approximates feature map of an RBF kernel) is used.</sup></sup><br>
<sup><sup>(5): Fast-slow LSTM with VAE like "variational unit" (VU) is used.</sup></sup><br>

---

## misc folder

The misc folder contains related example codes that I have put together while
learning RL. See the [README.md](https://github.com/ChuaCheowHuan/reinforcement_learning/tree/master/misc/README.md)
in the misc folder for more details.

---

## Blog

Check out my [blog](https://ChuaCheowHuan.github.io/) for more information on
my repositories.
