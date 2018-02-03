import tensorflow as tf

class BasicAgent():
    '''This agent uses a 1-hidden-layer dense NN to compute probability of going UP'''
    def __init__(self):

        def weight_variable(shape):
        	 initial = tf.truncated_normal(shape, stddev=0.05)
        	 return tf.Variable(initial)
          
        self.W1 = weight_variable([80*80, hidden_size_1])
        self.W2 = weight_variable([hidden_size_1, 1])
        self.input_vectors = tf.placeholder(shape=(None, 80*80), dtype=tf.float32)
        self.actions       = tf.placeholder(shape=(None,1),      dtype=tf.float32)
        self.rewards       = tf.placeholder(shape=(None,1),      dtype=tf.float32)
        
        self.hidden_layer = tf.nn.relu(tf.matmul(self.input_vectors, self.W1))
        self.output_layer = tf.nn.sigmoid(tf.matmul(self.hidden_layer, self.W2))
        
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output_layer, labels=self.rewards))
        
        self.GD = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    
    
     
    def action(self, sess, diff_frame):
        '''returns a probability of going UP at this frame'''
        feed_dict = {self.input_vectors:diff_frame}
        predicted_action = sess.run(self.output_layer, feed_dict=feed_dict)[0,0]
        action = 3 + np.random.binomial(1, predicted_action)
        return action
    
    def gym_action(self, sess, diff_frame):
        return 3 + self.action(self, sess, diff_frame)
    	                                                           
    def train(self, sess, diff_frames, actions, wins):
        '''trains the agent on the data'''
        feed_dict={self.input_vector:diff_frames, self.actions:actions, self.rewards:wins}
        _, loss = sess.run([self.GD.self.minimize(self.loss), self.loss], feed_dict=feed_dict)
        return loss
