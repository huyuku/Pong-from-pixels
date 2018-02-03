'''
This module is used to generate data by having the agent play pong.
'''
import tensorflow as tf
import random
import numpy as np
import agents
import gym
import cProfile
from config import *

#define preprocessing functions.
def preprocess(frame):
	""" preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
	frame = frame[35:195] # crop
	frame = frame[::2,::2,0] # downsample by factor of 2
	frame[frame == 144] = 0 # erase background (background type 1)
	frame[frame == 109] = 0 # erase background (background type 2)
	frame[frame != 0] = 1 # everything else (paddles, ball) just set to 1
	return frame.astype(np.float).ravel()

def ex(preprocessed_frames):
    '''expand array along first dimension, used if running the network with batch_size=1'''
    return np.expand_dims(preprocessed_frames,0)

def diff_frame(ordered_frames):
	'''compares frames from the current and last timestep, in order to be able to capture motion'''
	return preprocess(ordered_frames[0] - ordered_frames[1])

class train_set():
	'''Storing self-play games for training'''
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
    OpenAI_bot_score = 0
    agent_score = 0
    frame_buffer = [0,0] # "Working memory" of recent frames

    for m in range(2):
        env.render()
        frame_buffer[m], r, done, i = env.step(env.action_space.sample()) # take a random action

    t1 = time.time()
    while not done:
        env.render()
        if without_net:
            a = env.action_space.sample()
        else:
            a = agent.gym_action(session, ex(diff_frame(frame_buffer)))
        #Update the short term memory
        frame_buffer[0]=frame_buffer[1]
        frame_buffer[1], r, done, i = env.step(a) #Converting between a binary action representation, used in the loss function, and the gym representation (3-4)
        #Add the state action pair (s, a) to the temporary history, later to be added to the
        #main train data object once we have a reward r, which gives the complete data-point (s, a, r)
        temp_history.extend([[diff_frame(frame_buffer), a-3]])
        if abs(r) == 1:
			# Update the scores
            if r == 1:
                agent_score += 1
            else:
                OpenAI_bot_score += 1

            if quick_self_play: #Whether to play first to 20 wins or just first to 1
                done = True
            else:
                for entry in temp_history: #Update main train data-set
                    train_data.add(entry[0], entry[1], r)

                temp_history = []
        t2 = time.time()


    env.render(close=True)
    return OpenAI_bot_score, agent_score

#define preprocessing functions
def preprocess(frame):
	""" preprocess 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
	frame = frame[35:195] # crop
	frame = frame[::2,::2,0] # downsample by factor of 2
	frame[frame == 144] = 0 # erase background (background type 1)
	frame[frame == 109] = 0 # erase background (background type 2)
	frame[frame != 0] = 1 # everything else (paddles, ball) just set to 1
	return frame.astype(np.float).ravel()

def ex(preprocessed_frames):
    '''expand array along first dimension, used if running the network with batch size = 1'''
    return np.expand_dims(preprocessed_frames,0)

def diff_frame(ordered_frames):
	'''compares frames from the current and last timestep, in order to be able to capture motion'''
	return preprocess(ordered_frames[0] - ordered_frames[1])

agent = agents.BasicAgent(hidden_size, learning_rate)
env = gym.make('Pong-v0')

def main_function():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_iterations):
            print("Iteration %s. Resetting dataset" % i)
            train_data = train_set()

            print("Starting self-play...")
            t1 = time.time()
            for n in range(num_self_play_games):
                if n%100 == 0:
                    t2 = time.time()
                    print("Self-play game: %s" %n)
                    if print_analytics:
                        print("Self-play time: " + str(t2-t1) + " seconds.")
                        t1 = time.time()
                self_play(sess, agent, env, train_data)

            print("Starting training...")
            t1 = time.time()
            for e in range(num_train_epochs):
                diff_frames, actions, wins = train_data.sample(train_batch_size)
                loss = agent.train(sess, diff_frames, actions, wins)
                print("Loss epoch %s: %s" % (e, loss))
				t2 = time.time()
				if print_analytics:
					print("Train epoch time: " + str(t2-t1) + " seconds.")
				t1 = t2

if print_analytics:
	cProfile.run('main_function()') #Probably not very helpful
else:
	main_function()
