import tensorflow as tf
import numpy as np
import threading
import multiprocessing
import gym
from time import sleep
import agents
import parallelisation_tools as pll
import game
import data
import logging
import logging_agent
from game import *
from config import *
from worker import *

PARALLEL = True

@pll.Coordinator( PARALLEL )
def play(sess, dataset, agent, env, thread="worker_1", GAMES_PER_ITER=GAMES_PER_ITER):
    '''
    Plays games against itself, sequentially or in parallel.

    * comments:

    if you're changing it, make sure the first argument is still sess!
    '''
    for i_game in range(GAMES_PER_ITER):

        f, a, r, o_s, a_s = game.create_play_data(sess, agent, env, without_net=WITHOUT_NET, quick_play=QUICK_PLAY)
        dataset.add(f,a,r)
        if a_s>o_s:
            global wins
            wins += 1
        else:
            global losses
            losses += 1
        if i_game % 20 == 0:
            agent.playing_log(i_game, wins, losses)
            print("{0}. Game {1}. Agent: {2}, Opponent: {3}".format(thread, i_game, a_s, o_s))

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
    initial_agent = agents.BasicAgent(name, HIDDEN_SIZE, LEARNING_RATE)
    final_agent = logging_agent.Logging_Agent(initial_agent)
    return final_agent

agent = wrapped_agent('main_agent')
dataset = data.Dataset()
if PARALLEL:



env = gym.make('Pong-v0')
logging.basicConfig(filename='info.log',level=logging.INFO)
logger = debugtools.Logger()

def main_function():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        global wins
        global losses
        wins = 0
        losses = 0
        agent.set_time_start()
        for i in range(NUM_ITERATIONS):
            dataset.reset()
            print("Iteration {0}".format(i+1))
            print("Playing games:")
            play(sess, dataset, agent, env, GAMES_PER_ITER) # A bit unhappy with GAMES_PER_ITER happening inside here, instead of
                                                            # in a foor loop, but have no better solution atm


            print("Size of dataset: {0}".format(dataset.size))
            print("Training:")
            for epoch in range(EPOCHS_PER_ITER):

                f,a,r = dataset.sample(TRAIN_BATCH_SIZE)
                loss = agent.train(sess,f,a,r)
                for worker in WORKERS:
                    worker.update_local_ops
                if epoch % 500 == 0:
                    agent.epoch_log(sess, epoch, loss)
                    #print("Epoch {0}. Loss: {1}".format(epoch, loss))


if __name__ == "__main__":
    main_function()
