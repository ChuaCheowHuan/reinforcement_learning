# sample code snippet from:
# https://github.com/tensorflow/community/blob/master/rfcs/20181016-replicator.md

with strategy.scope():
  agent = Agent(num_actions, hidden_size, entropy_cost, baseline_cost)
  optimizer = tf.keras.optimizers.RMSprop(learning_rate)

# Queues of trajectories from actors.
queues = []
def learner_input(ctx):
  del ctx  # Unused.
  queue = tf.FIFOQueue(
      capacity=1, dtypes=trajectory_dtypes, shapes=trajectory_shapes)
  queues.append(queue)

  def dequeue_batch():
    batch = [Transition(*queue.dequeue()) for _ in range(batch_size_per_replica)]
    # Stack the `Tensor` nests along axis 1.
    return tf.nest.map_structure(lambda *xs: tf.stack(xs, axis=1), *batch)
  return dequeue_batch

def learner_step(trajectories):
  with tf.GradientTape() as tape:
    loss = tf.reduce_sum(agent.compute_loss(trajectories))

  agent_vars = agent.get_all_variables()
  grads = tape.gradient(loss, agent_vars)
  optimizer.apply_gradients(list(zip(grads, agent_vars)))
  return loss, agent_vars

# Create learner inputs.
learner_inputs = strategy.make_input_iterator(learner_input)

def run_actor(actor_id):
  queue = queues[actor_id % len(queues)]
  for _ in range(num_trajectories_per_actor):
    observation = get_observation_from_environment()
    action_taken, logits = agent(tf.expand_dims(observation, axis=0))
    trajectory = Transition(observation, action_taken, logits)
    queue.enqueue(tf.nest.flatten(trajectory))

# Start the actors.
for actor_id in range(num_actors):
  threading.Thread(target=run_actor, args=(actor_id,)).start()

# Run the learner.
strategy.initialize()

for _ in range(num_train_steps):
  per_replica_outputs = strategy.run(learner_step, learner_inputs)
  per_replica_losses, updated_agent_var_copies = zip(*per_replica_outputs)
  mean_loss = strategy.reduce(AggregationType.MEAN, per_replica_losses)

strategy.finalize()



# Global Batch Normalization
# When using a standard batch normalization layer with DistributionStrategy,
# the calculated mean and variance will be with-respect-to the local batch.
# A global batch normalization layer could be built using the all_reduce method.
def global_batch_norm(x):
  ctx = tf.distribute.get_replica_context()
  local_x_mean = tf.reduce_mean(x, axis=0)
  local_x_squared_mean = tf.reduce_mean(tf.square(x), axis=0)
  global_x_mean, global_x_squared_mean = (
      ctx.all_reduce([local_x_mean / ctx.num_replicas_in_sync,
                     local_x_squared_mean / ctx.num_replicas_in_sync], AggregationType.SUM)
  global_x_variance = global_x_squared_mean - tf.square(global_x_mean)
  return tf.nn.batch_normalization(
      x, global_x_mean, global_x_variance, offset=None, scale=None)
