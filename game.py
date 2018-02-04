'''
This module is used to generate data by having the agent play pong.
'''
import tensorflow as tf
import random
import numpy as np
import agents
import gym
#import cProfile
from config import *
import debugtools

#define preprocessing functions.
def preprocess(frame):
	""" preprocess 210x160x3 uint8 frame into 6400 float vector """
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
		#env.render()
		frame_buffer[m], r, done, i = env.step(env.action_space.sample()) # take a random action

	t1 = time.time()
	while not done:
		#env.render()
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
			# Update the score
			if r == 1:
				agent_score += 1
			else:
				OpenAI_bot_score += 1
			for entry in temp_history: #Update main train data-set
				train_data.add(entry[0], entry[1], r)
			if quick_self_play: #Whether to play first to 20 wins or just first to 1
				done = True
			else:
				temp_history = []
		t2 = time.time()

	#env.render(close=True)
	return OpenAI_bot_score, agent_score

def create_play_data(session, agent, env, without_net=False, quick_play=False):
	"""
	Generates data of the agent playing against the built-in pong AI.

	* arguments:
		session: the tf.Session used to run the agent.
		agent: the agent.
		env: the gym environment.
		without_net: if True, samples actions randomly. Used for testing.
		quick_play: plays games to the first point scored, instead of to 20.

	* returns:
		diff_frames, actions, rewards, opponent_score, agent_score

		diff_frames: a 6400x1 vector representing difference frames at each t.
		actions: The action taken at t. 1 if UP, 0 if down.
		rewards: 1 if t leads up to a goal, -1 otherwise.
		opponent_score, agent_score: number of goals scored in the episode.
	"""
	env.reset()
	done = False

	diff_frame_sets = []
	action_sets = []
	reward_sets = []

	opponent_score = 0
	agent_score = 0
	diff_frames  = []
	actions = []

	current_frame = 0
	current_action = env.action_space.sample() #take a random action to start with
	while not done:
		#make observation
		last_frame = current_frame
		f, r, done, i = env.step(current_action)
		current_frame = preprocess(f)
		diff_frame = current_frame - last_frame
		diff_frames.append(diff_frame)
		#take action
		if without_net:
			current_action = env.action_space.sample()
		else:
			current_action = agent.gym_action(session, np.expand_dims(diff_frame, axis=0))
		actions.append(current_action-3) # -3 needed to convert from gym representation to agent's representation

		if abs(r) == 1: #a goal is scored
			if r == 1:
				agent_score += 1
			else:
				opponent_score += 1

			if quick_play:
				done = True

			#package up the data for the frames involved in this goal
			diff_frame_sets.append(np.stack(diff_frames, axis=0))
			action_sets.append(np.stack(actions, axis=0))
			reward_sets.append(r * np.ones(len(actions)))
			diff_frames  = []
			actions = []

	#unpack data for each goal and combine into full sets
#	print("just before concatenating:", np.size(diff_frame_sets), len(diff_frame_sets))
#	print(diff_frame_sets)
	diff_frames_out = np.concatenate(diff_frame_sets, axis=0)
	actions_out = np.concatenate(action_sets, axis=0)
	rewards_out = np.concatenate(reward_sets, axis=0)

	return diff_frames_out, actions_out, rewards_out, opponent_score, agent_score

def test():
	with tf.Session() as sess:
		print("attempting to generate data...")
		agent = agents.BasicAgent()
		sess.run(tf.global_variables_initializer())
		env = gym.make('Pong-v0')
		f,a,r,o_s,a_s = create_play_data(sess, agent, env, quick_play=True)
		print("successfully generated data!")
		print("diff_frames shape:", f.shape)
		print("actions shape:", a.shape)
		print("rewards shape:", r.shape)

		print("attempting to train agent on data:")
		agent.train(sess, f, a, r)
		print("training step successful!")
		print("data is all good!")

if __name__ == "__main__":
	test()
