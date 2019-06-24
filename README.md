# reinforcement_learning

All Jupyter notebooks in this repo can be run directly in Google's Colab.

All results are displayed as output charts shown at the bottom of the notebooks.

The blog for this repo is available at the following Github page:

[https://ChuaCheowHuan.github.io/](https://ChuaCheowHuan.github.io/)

If you're unable to view the Jupyter notebooks on Github
(This is a known Github issue which happens randomly as of 20090624),
copy the code's URL & paste it in the search bar at [nbviewer](https://nbviewer.jupyter.org/)

Example: My 9) DPPO continuous [notebook](https://nbviewer.jupyter.org/github/ChuaCheowHuan/reinforcement_learning/blob/master/policy_gradient_based/DPPO_cont_GAE_dist_GPU.ipynb) at nbviewer.

1) DQN [code](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/DQN_variants/DQN/DQN_cartpole.ipynb)

>A Deep Q Network implementation in tensorflow with target network & random
experience replay. CartPole-v0 is used as the discrete environment. [write-ups](https://chuacheowhuan.github.io/DQN/)

2) DDQN [code](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/DQN_variants/DDQN/double_DQN_cartpole.ipynb)

>A Double Deep Q Network (DDQN) implementation in tensorflow with random
experience replay. [write-ups](https://chuacheowhuan.github.io/DDQN/)

3) Duelling DDQN [code](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/DQN_variants/duel_DDQN/duelling_DDQN_cartpole.ipynb)

>A Dueling Double Deep Q Network (Dueling DDQN) implementation in tensorflow
with random experience replay. CartPole-v0 is used as the discrete environment. [write-ups](https://chuacheowhuan.github.io/Duel_DDQN/)

4) Duelling DDQN with PER [code](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/DQN_variants/duel_DDQN_PER/duelling_DDQN_PER_cartpole.ipynb)

>A Dueling Double Deep Q Network with Priority Experience Replay
(Duel DDQN with PER) implementation in tensorflow. CartPole-v0 is used as the
discrete environment. [write-ups](https://chuacheowhuan.github.io/Duel_DDQN_with_PER/)

5) A3C discrete with N-step targets (missing terms are treated as zero) [code](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/policy_gradient_based/A3C/A3C_disc_miss.ipynb)

>A3C (Asynchronous Advantage Actor Critic) implementation with
Tensorflow. This is a multi-threaded version. CartPole-v0 is used as the
discrete environment. [write-ups](https://chuacheowhuan.github.io/A3C_disc_thread_nStep/)

6) A3C discrete with N-step targets (maximum possible terms are used) [code](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/policy_gradient_based/A3C/A3C_disc_max.ipynb)

>A3C (Asynchronous Advantage Actor Critic) implementation with
Tensorflow. This is a multi-threaded version. CartPole-v0 is used as the
discrete environment. [write-ups](https://chuacheowhuan.github.io/A3C_disc_thread_nStep/)

7) A3C continuous with N-step targets (maximum possible terms are used) [code](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/policy_gradient_based/A3C/A3C_cont_max.ipynb)

>A3C (Asynchronous Advantage Actor Critic) implementation with
Tensorflow. This is a multi-threaded version. Pendulum-v0 is used as for
continuous environment. [write-ups](https://chuacheowhuan.github.io/A3C_cont_thread_nStep/)

8) A3C discrete with N-step targets (maximum possible terms are used) [code](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/policy_gradient_based/A3C/A3C_disc_max_dist.ipynb)

>A3C (Asynchronous Advantage Actor Critic) implementation with
distributed Tensorflow and Python's multiprocessing package.
CartPole-v0 is used as the discrete environment. [write-ups pending...]()

9) DPPO continuous (normalized running rewards with GAE) implementation with
distributed Tensorflow and Python's multiprocessing package. [code](https://github.com/ChuaCheowHuan/reinforcement_learning/blob/master/policy_gradient_based/DPPO_cont_GAE_dist_GPU.ipynb)

>Pendulum-v0 is used as the continuous environment. [write-ups pending...]()
