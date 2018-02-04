'''
This module is used to generate data by having the agent play pong.
'''
import tensorflow as tf
import random
import numpy as np
import agents
import gym
from config import *
import debugtools

def preprocess(frame):
	""" preprocess 210x160x3 uint8 frame into 6400 float vector """
	frame = frame[35:195] # crop
	frame = frame[::2,::2,0] # downsample by factor of 2
	frame[frame == 144] = 0 # erase background (background type 1)
	frame[frame == 109] = 0 # erase background (background type 2)
	frame[frame != 0] = 1 # everything else (paddles, ball) just set to 1
	return frame.astype(np.float).ravel()

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
