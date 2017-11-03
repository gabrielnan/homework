import tensorflow as tf
import numpy as np
from bc import build_network
from load_policy import load_policy
import matplotlib.pyplot as plt


def dagger(envname, data_file, expert_file, max_timesteps=False, render=False,
           num_rollouts=20, num_train_iters=1000, batch_size=100,
           input_name='observations', target_name='actions'):
    # Load initial dataset
    data = np.load(data_file)
    input_data = data[input_name]
    target_data = data[target_name]
    input_shape = list(input_data.shape[1:])
    target_shape = list(target_data.shape[1:])

    # Loading expert policy
    policy_fn = load_policy(expert_file)

    with tf.Session() as sess:
        # Create network structure
        x, y_, y, _, train_step = build_network(input_shape, target_shape)
        sess.run(tf.global_variables_initializer())

        # Setup environment
        import gym
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        for i in range(num_rollouts):
            print('iter', i)

            # Training with current aggregated data
            train(sess, train_step, input_data, x, target_data, y_, num_train_iters,
                  batch_size)

            observations = []
            target_actions = []
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                # Expert
                target_action = policy_fn(obs[None, :])
                target_actions.append(target_action)

                # Learner
                feed_dict = {x: obs.reshape([1] + list(obs.shape))}
                learner_action = sess.run(y, feed_dict=feed_dict)

                observations.append(obs)
                obs, r, done, _ = env.step(learner_action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            input_data = np.append(input_data, observations, axis=0)
            target_data = np.append(target_data, target_actions, axis=0)
            input_data, target_data = shuffle(input_data, target_data)
            returns.append(totalr)

        plt.xlabel('DAgger iterations')
        plt.ylabel('Returns')
        plt.plot(np.arange(num_rollouts), returns)
        plt.show()


def train(sess, train_step, input_data, x, target_data, y_, num_iters, batch_size):
    # training
    n = input_data.shape[0]
    for step in range(num_iters):
        batch_inds = np.mod(step * batch_size + np.arange(batch_size), n)
        feed_dict = {x: input_data[batch_inds], y_: target_data[batch_inds]}
        sess.run(train_step, feed_dict=feed_dict)


def shuffle(input_data, target_data):
    order = np.arange(input_data.shape[0])
    np.random.shuffle(order)
    return input_data[order], target_data[order]

if __name__ == '__main__':
    envname = 'Hopper-v1'
    data_file = 'data/data_Hopper-v1_rollouts_100.npz'
    expert_file = 'experts/Hopper-v1.pkl'

    dagger(envname, data_file, expert_file)
