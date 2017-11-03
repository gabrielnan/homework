import tensorflow as tf
import numpy as np


def run_model(model_file, envname, max_timesteps=False, num_rollouts=20,
              render=False):
    # load model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # tf_util.initialize()

        saver = tf.train.import_meta_graph(model_file + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('models/./'))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('input:0')
        y = graph.get_tensor_by_name('output:0')

        import gym
        env = gym.make(envname)
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        for i in range(num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                feed_dict = {x: obs.reshape([1] + list(obs.shape))}
                action = sess.run(y, feed_dict=feed_dict)[0]
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if render:
                    env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


if __name__ == '__main__':
    envname = 'Humanoid-v1'
    model_type = 'dagger'
    model_file = 'models/' + envname + '_' + model_type
    run_model(model_file, envname, render=True)
