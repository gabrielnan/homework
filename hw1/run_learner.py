import tensorflow as tf
import numpy as np
import load_policy


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', type=str,
                        default='models/Hopper-v1_model')
    parser.add_argument('--envname', type=str, default='Hopper-v1')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20)
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()

    policy_fn = load_policy.load_policy('experts/Hopper-v1.pkl')

    # load model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # tf_util.initialize()

        saver = tf.train.import_meta_graph(args.model_file + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('models/./'))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name('input:0')
        y = graph.get_tensor_by_name('output:0')

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                feed_dict = {x: obs.reshape([1] + list(obs.shape))}
                action = sess.run(y, feed_dict=feed_dict)[0]
                expert_action = policy_fn(obs[None, :])
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))


if __name__ == '__main__':
    main()
