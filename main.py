import tensorflow as tf
import numpy as np
import random
import gym
import cProfile
import agents
import game
from config import *
import time

agent = agents.BasicAgent(hidden_size, learning_rate)
env = gym.make('Pong-v0')

def main_function():
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(num_iterations):

			print("Iteration %s. Resetting dataset" % i)
			train_data = game.train_set()

			print("Starting self-play...")

			t1 = time.time()

			for n in range(num_self_play_games):
				if n%100 == 0:
					print("Self-play game: %s" %n)

				self_play(sess, agent, env, train_data)

			t2 = time.time()
			if print_analytics:
				print("Time: " + str(t1-t2) + " seconds.")

			print("Starting training...")
			for e in range(num_train_epochs):
				t1 = time.time()
				diff_frames, actions, wins = train_data.sample(train_batch_size)
				loss = agent.train(sess, diff_frames, actions, wins)
				print("Loss epoch %s: %s" % (e, loss))
<<<<<<< HEAD
				t2 = time.time()
				if print_analytics:
					print("Time: " + str(t1-t2) + " seconds.")
=======
if print_analytics:
	cProfile.run('main_function()') #Probably not very helpful
else:
	main_function()
>>>>>>> b294f0e5155a3e8caa286185bace17132811a4b7
