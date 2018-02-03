import tensorflow as tf
import numpy as np
import random
import gym
import cProfile
import agents
import game

#HYPERPARAMETERS
num_self_play_games = 50000
num_train_epochs = 100 #Change to something plausible later
train_batch_size = 20
num_iterations = 1
learning_rate = 0.01
hidden_size = 100
without_net = False
quick_self_play = True #For testing

agent = agents.BasicAgent(hidden_size, learning_rate)
env = gym.make('Pong-v0')


def main_function():
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(num_iterations):
			print("Iteration %s. Resetting dataset" % i)
			train_data = game.train_set()

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
