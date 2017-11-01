import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_filename', type=str, default='../data/data_Hopper-v1_rollouts_20.npz')
    parser.add_argument('--input', type=str, default='observations')
    parser.add_argument('--target', type=str, default='actions')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--nb_iters', type=int, default=10000)
    args = parser.parse_args()

    print('loading data')
    data = np.load(args.data_filename)
    x_data = data[args.input]
    y_data = data[args.target]
    n = x_data.shape[0]
    batch_size = args.batch_size

    print('building network')
    sess = tf.InteractiveSession()

    x, y_, loss, train_step = build_network(list(x_data.shape[1:]), list(y_data.shape[1:]))
    sess.run(tf.global_variables_initializer())
    losses = []

    for i in range(args.nb_iters):
        batch_inds = np.mod(batch_size + np.arange(batch_size), n)
        feed_dict = {x:x_data[batch_inds], y_:y_data[batch_inds]}
        loss_val = loss.eval(feed_dict=feed_dict)
        losses.append(loss_val)
        train_step.run(feed_dict=feed_dict)
        # print('iter:', i, '| Loss:', loss_val)
        

    # Plot losses
    plt.xlabel('Iterations')
    plt.ylabel('Mean Squared Loss')
    plt.plot(np.arange(args.nb_iters), losses)
    plt.show()


def build_network(x_shape, y_shape):
    x = tf.placeholder(tf.float32, shape=[None] + x_shape)
    y_ = tf.placeholder(tf.float32, shape=[None] + y_shape)

    W = tf.Variable(tf.zeros([np.product(x_shape), np.product(y_shape)]))
    b = tf.Variable(tf.zeros([np.product(y_shape)]))

    y = tf.reshape(tf.matmul(x, W) + b, [-1] + y_shape)

    loss = tf.reduce_mean(tf.square(y - y_))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    return x, y_, loss, train_step


if __name__ == '__main__':
    main()
