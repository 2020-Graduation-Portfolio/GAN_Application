import tensorflow as tf
import numpy as np

def encoder(self, inputs):
    with tf.variable_scope("encoder"):
        with tf.variable_scope("conv1"):
            c1 = tf.layers.conv2d(inputs, 32, [4, 4], [2, 2], 'same')
            c1 = tf.nn.leaky_relu(c1)
        with tf.variable_scope("conv2"):
            c2 = tf.layers.conv2d(c1, 64, [4, 4], [2, 2], 'same')
            c2 = tf.nn.leaky_relu(c2)
        with tf.variable_scope("conv3"):
            c3 = tf.layers.conv2d(c2, 128, [4, 4], [2, 2], 'same')
            c3 = tf.nn.leaky_relu(c3)
        with tf.variable_scope("conv4"):
            c4 = tf.layers.conv2d(c3, 256, [4, 4], [2, 2], 'same')
            c4 = tf.nn.leaky_relu(c4)
        with tf.variable_scope("conv5"):
            c5 = tf.layers.conv2d(c4, 512, [4, 4], [2, 2], 'same')
            c5 = tf.nn.leaky_relu(c5)
        with tf.variable_scope("fc"):
            fc1 = tf.reshape(c5, [-1, c5.shape[1] * c5.shape[2] * c5.shape[3]])
            fc2 = tf.reshape(c5, [-1, c5.shape[1] * c5.shape[2] * c5.shape[3]])
            z_mean = tf.layers.dense(fc1, 128, activation="softplus")
            z_log_var = tf.layers.dense(fc2, 128, activation="softplus")
    return z_avg, z_log_var
def decoder(self, inputs):
    inputs = tf.reshape(inputs, [-1, 1, 1, 152])
    with tf.variable_scope('decoder'):
        with tf.variable_scope("deconv1"):
            dc1 = tf.layers.conv2d_transpose(inputs, 512, (4, 4), (2, 2))
            dc1 = tf.layers.batch_normalization(x, training=True)
            dc1 = tf.nn.relu(dc1)
        with tf.variable_scope("deconv2"):
            dc2 = tf.layers.conv2d_transpose(dc1, 256, (4, 4), (2, 2))
            dc2 = tf.layers.batch_normalization(dc2, training=True)
            dc2 = tf.nn.relu(dc2)
        with tf.variable_scope("deconv3"):
            dc3 = tf.layers.conv2d_transpose(dc2, 128, (4, 4), (2, 2))
            dc3 = tf.layers.batch_normalization(dc3, training=True)
            dc3 = tf.nn.relu(dc3)
        with tf.variable_scope("deconv4"):
            dc4 = tf.layers.conv2d_transpose(dc3, 64, (4, 4), (2, 2))
            dc4 = tf.layers.batch_normalization(dc4, training=True)
            dc4 = tf.nn.relu(dc4)
        with tf.variable_scope("deconv5"):
            dc5 = tf.layers.conv2d_transpose(dc4, 32, (4, 4), (2, 2))
            dc5 = tf.layers.batch_normalization(dc5, training=True)
            dc5 = tf.nn.relu(dc5)
        with tf.variable_scope("deconv6"):
            dc6 = tf.layers.conv2d_transpose(dc5, 3, (3, 3), (1, 1))
            dc6 = tf.layers.batch_normalization(dc6, training=True)
            dc6 = tf.nn.sigmoid(dc6)
    return dc6