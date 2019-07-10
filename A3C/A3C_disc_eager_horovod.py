# issue:
# Only working for rank 0 (1 process),
# multiple processes causes error.

# Possible causes:
# "With Horovod, each rank must execute exactly the same number of iterations."
# "need to stick to a fixed number of iterations for every rank"

# horovod issue:
# https://github.com/horovod/horovod/issues/346

# command:
# python a3c_cartpole.py — algorithm=random — max-eps=10

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # use "" for CPU only

#import threading
import time
import gym
#import multiprocessing
import numpy as np
#from queue import Queue
import argparse
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras import layers

import horovod.tensorflow as hvd

tf.enable_eager_execution()

parser = argparse.ArgumentParser(description='Run A3C algorithm on the game '
                                             'Cartpole.')
parser.add_argument('--algorithm', default='a3c', type=str,
                    help='Choose between \'a3c\' and \'random\'.')
#parser.add_argument('--train', dest='train', action='store_true', # to play, store_true deaults to false
parser.add_argument('--train', dest='train', action='store_false', # to train, store_true deaults to true
                    help='Train our model.')
parser.add_argument('--lr', default=0.001,
                    help='Learning rate for the shared optimizer.')
parser.add_argument('--update-freq', default=20, type=int,
                    help='How often to update the global model.')
parser.add_argument('--max-eps', default=2, type=int,
                    help='Global maximum number of episodes to run.')
parser.add_argument('--gamma', default=0.99,
                    help='Discount factor of rewards.')
parser.add_argument('--save-dir', default='./tmp/', type=str,
                    help='Directory in which you desire to save the model.')
#args = parser.parse_args() # original line
args = parser.parse_args(args=[]) # switch to this line to work in Colab


class Memory:
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []

  def store(self, state, action, reward):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)

  def clear(self):
    self.states = []
    self.actions = []
    self.rewards = []


class ActorCriticModel(keras.Model):
  def __init__(self, state_size, action_size):
    super(ActorCriticModel, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.dense1 = layers.Dense(100, activation='relu')
    self.policy_logits = layers.Dense(action_size)
    self.dense2 = layers.Dense(100, activation='relu')
    self.values = layers.Dense(1)

  def call(self, inputs):
    # Forward pass
    x = self.dense1(inputs)
    logits = self.policy_logits(x)
    v1 = self.dense2(inputs)
    values = self.values(v1)
    return logits, values


def compute_loss(AC_model, done, new_state, memory, gamma=0.99):
    if done:
        reward_sum = 0.  # terminal
    else:
        reward_sum = AC_model(tf.convert_to_tensor(new_state[None, :], dtype=tf.float32))[-1].numpy()[0]

    # Get discounted rewards
    discounted_rewards = []
    for reward in memory.rewards[::-1]:  # reverse buffer r
      reward_sum = reward + gamma * reward_sum
      discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()

    logits, values = AC_model(tf.convert_to_tensor(np.vstack(memory.states), dtype=tf.float32))
    # Get our advantages
    advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None], dtype=tf.float32) - values
    # Value loss
    value_loss = advantage ** 2 # ** means squared

    # Calculate our policy loss
    policy = tf.nn.softmax(logits)
    entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=policy, logits=logits)

    policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions, logits=logits)
    policy_loss *= tf.stop_gradient(advantage)
    policy_loss -= 0.01 * entropy
    total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
    return total_loss


def main(_):
    # Horovod: initialize Horovod.
    hvd.init()
    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    tf.enable_eager_execution(config=config)

    if hvd.rank() == 0:
        save_dir = args.save_dir
        save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    game_name = 'CartPole-v0'
    env = gym.make(game_name)
    #env = gym.make(game_name).unwrapped
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    print(state_size, action_size)

    AC_model = ActorCriticModel(state_size, action_size)
    AC_model(tf.convert_to_tensor(np.random.random((1, state_size)), dtype=tf.float32))

    # Horovod: adjust learning rate based on number of GPUs.
    #opt = tf.train.RMSPropOptimizer(0.001 * hvd.size())
    opt = tf.train.AdamOptimizer(args.lr * hvd.size(), use_locking=True)

    if hvd.rank() == 0:
        """
        # saving
        checkpoint_dir = './checkpoints'
        step_counter = tf.train.get_or_create_global_step()
        checkpoint = tf.train.Checkpoint(model=AC_model, optimizer=opt, step_counter=step_counter)
        """
        moving_average_rewards = []  # record episode reward to plot

# training eps loop
# **************************************************
    total_step = 1
    mem = Memory()
    global_episode = 0
    while global_episode < args.max_eps: # all eps
      current_state = env.reset()
      mem.clear()
      ep_reward = 0.
      ep_steps = 0
      ep_loss = 0

      time_count = 0 # t
      done = False
      while not done: # per eps
        logits, _ = AC_model(tf.convert_to_tensor(current_state[None, :], dtype=tf.float32))
        probs = tf.nn.softmax(logits)

        action = np.random.choice(action_size, p=probs.numpy()[0])
        new_state, reward, done, _ = env.step(action)
        if done:
          reward = -1
          if hvd.rank() == 0:
              moving_average_rewards.append(ep_reward)

        ep_reward += reward
        mem.store(current_state, action, reward)

        if time_count == args.update_freq or done: # training
          print(hvd.rank(), global_episode, time_count, done, ep_reward)

          # Calculate gradient wrt to local model. We do so by tracking the
          # variables involved in computing the loss by using tf.GradientTape
          with tf.GradientTape() as tape:
            total_loss = compute_loss(AC_model, done, new_state, mem, args.gamma)
          ep_loss += total_loss

          # Horovod: broadcast initial variable states from rank 0 to all other processes.
          # This is necessary to ensure consistent initialization of all workers when
          # training is started with random weights or restored from a checkpoint.
          """
          if global_episode == 0:
              hvd.broadcast_variables(AC_model.variables, root_rank=0)
          """
          # Horovod: add Horovod Distributed GradientTape.
          tape = hvd.DistributedGradientTape(tape)

          # Calculate local gradients
          grads = tape.gradient(total_loss, AC_model.trainable_weights)
          # Push local gradients to global model
          opt.apply_gradients(zip(grads, AC_model.trainable_weights), global_step=tf.train.get_or_create_global_step())
          # Update local model with new weights
          #self.local_model.set_weights(self.global_model.get_weights())

          mem.clear()
          time_count = 0

        ep_steps += 1
        time_count += 1
        current_state = new_state
        total_step += 1

      global_episode += 1
# **************************************************
# training eps while loop ends

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting it.
    if hvd.rank() == 0:

        print('global_episode', global_episode)

        """checkpoint.save(checkpoint_dir)"""

        plt.plot(moving_average_rewards)
        plt.ylabel('Moving average ep reward')
        plt.xlabel('Step')
        plt.savefig(os.path.join(save_dir, '{} Moving Average.png'.format(game_name)))
        #plt.show()

#start_time = time.time()

if __name__ == "__main__":
    tf.app.run()

#print("--- %s seconds ---" % (time.time() - start_time))
