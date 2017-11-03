import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def bc(data_file, model_file, input_name='observations', target_name='actions',
       batch_size=100, nb_iters=10000, plot=True):
    print('loading data')
    data = np.load(data_file)
    x_data = data[input_name]
    y_data = data[target_name]
    n = x_data.shape[0]
    batch_size = batch_size

    print('building network')
    sess = tf.InteractiveSession()

    x, y_, _, loss, train_step = build_network(list(x_data.shape[1:]),
                                            list(y_data.shape[1:]))
    sess.run(tf.global_variables_initializer())
    losses = []

    saver = tf.train.Saver()

    for step in range(nb_iters):
        batch_inds = np.mod(step * batch_size + np.arange(batch_size), n)
        feed_dict = {x: x_data[batch_inds], y_: y_data[batch_inds]}
        loss_val = loss.eval(feed_dict=feed_dict)
        losses.append(loss_val)
        train_step.run(feed_dict=feed_dict)

    saver.save(sess, model_file)

    # Plot losses
    if plot:
        plt.xlabel('Iterations')
        plt.ylabel('Mean Squared Loss')
        plt.plot(np.arange(nb_iters), losses)
        plt.show()


def build_network(x_shape, y_shape):
    x = tf.placeholder(tf.float32, shape=[None] + x_shape, name='input')
    y_ = tf.placeholder(tf.float32, shape=[None] + y_shape, name='target')

    dense1 = tf.layers.dense(inputs=x, units=256, activation=tf.nn.relu,
                             use_bias=True)
    dense2 = tf.layers.dense(inputs=dense1, units=512, activation=tf.nn.relu,
                             use_bias=True)
    dense3 = tf.layers.dense(inputs=dense2, units=256, activation=tf.nn.relu,
                             use_bias=True)
    dense4 = tf.layers.dense(inputs=dense3, units=np.product(y_shape),
                             activation=None, use_bias=True)

    # W = tf.Variable(tf.zeros([np.product(x_shape), np.product(y_shape)]))
    # b = tf.Variable(tf.zeros([np.product(y_shape)]))

    y = tf.reshape(dense4, [-1] + y_shape, name='output')

    loss = tf.reduce_mean(tf.square(y - y_))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    return x, y_, y, loss, train_step

if __name__ == '__main__':
    envname = 'Humanoid-v1'
    data_rolls = 100
    data_file = 'data/' + envname + '_data_' + str(data_rolls) + '.npz'
    model_file = 'models/' + envname + '_model'

    bc(data_file, model_file)
