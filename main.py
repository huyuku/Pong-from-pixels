import tensorflow as tf
import numpy as np
import random
import gym
#import cProfile
import agents
import game
import logging
from game import *
from config import *

agent = agents.BasicAgent(hidden_size, learning_rate)
env = gym.make('Pong-v0')
logging.basicConfig(filename='info.log',level=logging.INFO)
logger = debugtools.Logger()

def old_main_function():
    wins = 0
    losses = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_iterations):
            print("Iteration %s. Resetting dataset" % i)
            train_data = game.train_set()

            print("Starting self-play...")
            timer.set_time_start()
            for n in range(num_self_play_games):
                if n%10 == 0:
                    print("Self-play game: %s" %n)
                    logger.loginfo("Score after %s games: %s wins, %s losses." % (n, wins, losses))
                    logger.logtime('10 self-play games')
                OpenAI_score, agent_score = game.self_play(sess, agent, env, train_data)
                if agent_score>OpenAI_score:
                    wins += 1
                else:
                    losses += 1

            print("Starting training...")
            timer.set_time_start()
            for e in range(num_train_epochs):
                diff_frames, actions, wins = train_data.sample(train_batch_size)
                loss = agent.train(sess, diff_frames, actions, wins)
                logger.loginfo("Loss epoch %s loss: %s" % (e, loss))
                logger.logtime('Train epoch')

def main_function():
    with tf.Session as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(num_iterations):
            f, a, r, o_s, a_s = game.create_play_data(sess, agent, env)
            print("Game {0}. Agent score: {1}, Opponent_score: {2}".format(i,a_s,o_s))
            agent.train(sess,f,a,r)

if __name__ == "__main__":
    main_function()
