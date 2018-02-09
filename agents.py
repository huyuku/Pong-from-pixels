import tensorflow as tf
import numpy as np
import debugtools
import logging_agent
from config import HIDDEN_SIZE, LEARNING_RATE

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def wrapped_agent(name):
    '''
    Wrapper used to make it easier to modify agents using modules (e.g. the logging module),
    without there being multiple other dependencies in main.py that have to be changed should
    an additional wrapper be introduced on the agent

    * arguments:

    name
        a str used as a prefix (or scope) for the tensorflow variables

    * comments:

    N.A.

    '''
    initial_agent = BasicAgent(name, HIDDEN_SIZE, LEARNING_RATE)
    final_agent = logging_agent.Logging_Agent(initial_agent)
    return final_agent

class BasicAgent():
    '''
    Uses a 1-hidden-layer dense NN to compute probability of going UP,
    then samples using that probability to decide its action.

    * arguments:

    hidden_size, default=100
        controls the number of nodes in the hidden layer.

    learning_rate, default=0.01
        controls the learning rate of the optimiser used for training.

    * comments:

    uses tf.AdamOptimiser for its training step.

    '''
    def __init__(self, scope, hidden_size=100, learning_rate=0.01):
        with tf.variable_scope(scope):
            def weight_variable(shape, name):
                initial = tf.truncated_normal(shape, stddev=0.05)
                return tf.Variable(initial, name=name)

            self.W1 = weight_variable([80*80, hidden_size], "W1")
            self.W2 = weight_variable([hidden_size, 1], "W2")

            self.frames  = tf.placeholder(shape=(None, 80*80), dtype=tf.float32, name="frames_in")  # flattened diff_frame
            self.actions = tf.placeholder(shape=(None,), dtype=tf.float32, name="action_in")  # 1 if agent went UP, 0 otherwise
            self.rewards = tf.placeholder(shape=(None,), dtype=tf.float32, name="reward_in")  # 1 if frame comes from a won game, -1 otherwise

            self.hidden_layer = tf.nn.relu(tf.matmul(self.frames, self.W1), name="hidden_layer")
            self.output_layer = tf.nn.sigmoid(tf.matmul(self.hidden_layer, self.W2), name="output_layer")

            # loss = - sum over i of reward_i * logp(action_i | frame_i)
            self.loss = -tf.reduce_mean(self.rewards * (self.actions * self.output_layer + (1-self.actions) * (1-self.output_layer)),
                                        name="loss")

            self.Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_step = self.Optimizer.minimize(self.loss)

    def action(self, sess, diff_frame):
        '''returns a probability of going UP at this frame'''
        feed_dict = {self.frames:diff_frame}
        predicted_action = sess.run(self.output_layer, feed_dict=feed_dict)[0,0]
        action = np.random.binomial(1, predicted_action)
        return action

    def gym_action(self, sess, diff_frame):
        return 3 + self.action(sess, diff_frame)

    def train(self, sess, diff_frames, actions, rewards):
        '''trains the agent on the data'''
        feed_dict={self.frames:diff_frames, self.actions:actions, self.rewards:rewards}
        _, loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
        return loss

    def set_time_start(self):
        return

class ConvNetAgent():
    '''
    Uses a ConvNet to compute the probabilty of going UP,
    then samples using that probability to decide its action.

    * arguments:

    hidden_size, default=100
        controls the number of nodes in the hidden layer.

    learning_rate, default=0.01
        controls the learning rate of the optimiser used for training.

    * comments:

    uses tf.AdamOptimiser for its training step.

    '''
    def __init__(self, scope,
                 channels_num=30,
                 connected_size=100,
                 learning_rate=0.01,
                 dropout_p=0.5
                 ):
        with tf.variable_scope(scope):

            self.frames_in = tf.placeholder(shape=(None, 80*80), dtype=tf.float32, name="frames_in")  # flattened diff_frame
            self.actions = tf.placeholder(shape=(None,), dtype=tf.float32, name="action_in")  # 1 if agent went UP, 0 otherwise
            self.rewards = tf.placeholder(shape=(None,), dtype=tf.float32, name="reward_in")  # 1 if frame comes from a won game, -1 otherwise

            self.frames = tf.reshape(self.frames_in, [-1, 80, 80, 1])

            self.conv1 = tf.layers.conv2d(inputs=self.frames,
                                          filters=channels_num,
                                          kernel_size=[5,5],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          name="conv1")

            self.pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                                 pool_size=[2,2],
                                                 strides=2,
                                                 name='pool1')

            self.conv2 = tf.layers.conv2d(inputs=self.pool1,
                                          filters=channels_num,
                                          kernel_size=[5,5],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          name="conv2")

            self.pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                                 pool_size=[2,2],
                                                 strides=2,
                                                 name='pool2')

            self.pool2flat = tf.reshape(self.pool2flat, [-1, 20*20*channels_num])

            self.fully_connected = tf.layers.dense(inputs=self.pool2flat,
                                                   units=connected_size,
                                                   activation=tf.nn.relu,
                                                   name='fully_connected')

            self.dropout = tf.layers.dropout(inputs=self.fully_connected,
                                             rate=dropout_p,
                                             name='dropout')

            self.output_layer = tf.layers.dense(inputs=self.dropout,
                                                units=1,
                                                activation=tf.nn.sigmoid,
                                                name='out')

            # loss = - sum over i of reward_i * logp(action_i | frame_i)
            self.loss = -tf.reduce_mean(self.rewards * (self.actions * self.output_layer + (1-self.actions) * (1-self.output_layer)),
                                        name="loss")

            self.Optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_step = self.Optimizer.minimize(self.loss)

    def action(self, sess, diff_frame):
        '''returns a probability of going UP at this frame'''
        feed_dict = {self.frames:diff_frame}
        predicted_action = sess.run(self.output_layer, feed_dict=feed_dict)[0,0]
        action = np.random.binomial(1, predicted_action)
        return action

    def gym_action(self, sess, diff_frame):
        return 3 + self.action(sess, diff_frame)

    def train(self, sess, diff_frames, actions, rewards):
        '''trains the agent on the data'''
        feed_dict={self.frames:diff_frames, self.actions:actions, self.rewards:rewards}
        _, loss = sess.run([self.train_step, self.loss], feed_dict=feed_dict)
        return loss

    def set_time_start(self):
        return
