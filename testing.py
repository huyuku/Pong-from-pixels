import tensorflow as tf
import numpy as np
import random
import gym
import numpy as np
import cProfile

#HYPERPARAMETERS
num_self_play_games = 50000
num_train_epochs = 100 #Change to something plausible later
train_batch_size = 20
num_iterations = 1
learning_rate = 0.01
hidden_size_1 = 100000

class train_set(): 
    '''Storing self-play games for training'''
    '''NOTE: CURRENTLY DOESN'T DEAL WITH DIFF FRAMES'''
    def __init__(self, size = 1000000):
        self.history = []
        self.size = size
        
    def add(self, s, a, r):
        if len(self.history) + 1 >= self.size:
            self.history[0:(1+len(self.history))-self.size] = []
        self.history.extend([[s, a, r]])
            
    def sample(self,batch_size):
        idxes = random.sample(range(len(self.history)), batch_size)
        s = np.stack([self.history[:][idx][0] for idx in idxes], 0)
        a = np.stack([self.history[:][idx][1] for idx in idxes], 0)
        r = np.expand_dims(np.stack([self.history[:][idx][2] for idx in idxes], 0), 1)
        return s, a, r
    
    
def self_play(session, agent, env, train_data):
    env.reset()
    done = False
    temp_history = []
    
    s = [0,0] #Storing the memory of recent frames
    for m in range(2):
        env.render()
        s[m], r, done, i = env.step(env.action_space.sample()) # take a random action
      
    while not done:
        env.render()
        a = agent.action(session, diff_frame(s))
        s[0]=s[1]
        s[1], r, done, i = env.step(a) 
        temp_history.extend([diff_frame(s), a]) # this might not happen on the last move
        
    for entry in temp_history:
        train_data.add(entry[0], entry[1], r)
    env.render(close=True)

#define preprocessing functions
def preprocess(frame):
  """ preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  frame = frame[35:195] # crop
  frame = frame[::2,::2,0] # downsample by factor of 2
  frame[frame == 144] = 0 # erase background (background type 1)
  frame[frame == 109] = 0 # erase background (background type 2)
  frame[frame != 0] = 1 # everything else (paddles, ball) just set to 1
  return frame.astype(np.float).ravel()

def diff_frame(ordered_frames):
	return np.expand_dims(preprocess(ordered_frames[0] - ordered_frames[1]),0)

#define the model
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

agent = BasicAgent()
env = gym.make('Pong-v0')

cProfile.run('mainfunction()')
 
def mainfunction():
	with tf.Session() as sess:	
	    sess.run(tf.global_variables_initializer()) 
	    for i in range(num_iterations):
		print("Iteration %s. Resetting dataset" % i)
		train_data = train_set()

		print("Starting self-play...")
		for n in range(num_self_play_games):
		  if n%100 == 0:
		    print("Self-play game: %s" %n)

		  self_play(sess, agent, env, train_data)
		print("Starting training...")
		for e in range(num_train_epochs):
		  diff_frames, actions, wins = train_data.sample(train_batch_size)
		  loss = agent.train(sess, diff_frames, actions, wins)
		  print("Loss epoch %s: %s" % (e, loss))
                               
                               
 
