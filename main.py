import tensorflow as tf
import numpy as np
import random
import gym
#import cProfile
import agents
import game
from game import *
from config import *
import time
import logging

agent = agents.BasicAgent(hidden_size, learning_rate)
env = gym.make('Pong-v0')
logging.basicConfig(filename='info.log',level=logging.INFO)

agent = agents.BasicAgent(hidden_size, learning_rate)
env = gym.make('Pong-v0')

def main_function():
    wins = 0
    losses = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_iterations):
            print("Iteration %s. Resetting dataset" % i)
            train_data = game.train_set()

            print("Starting self-play...")
            t1 = time.time()
            for n in range(num_self_play_games):
                if n%10 == 0:
                    t2 = time.time()
                    print("Self-play game: %s" %n)
                    print("Current score: %s wins, %s losses." % (wins, losses))
                    if print_analytics:
                        print("Self-play time: %s seconds." % (t2-t1))
                        t1 = time.time()
                OpenAI_score, agent_score = game.self_play(sess, agent, env, train_data)
                if agent_score>OpenAI_score:
                    wins += 1
                else:
                    losses += 1

            print("Starting training...")
            t1 = time.time()
            for e in range(num_train_epochs):
                diff_frames, actions, wins = train_data.sample(train_batch_size)
                loss = agent.train(sess, diff_frames, actions, wins)
                print("Loss epoch %s: %s" % (e, loss))
                t2 = time.time()
                if print_analytics:
                    print("Train epoch time: %s seconds." % (t2-t1))
                t1 = t2

main_function()
