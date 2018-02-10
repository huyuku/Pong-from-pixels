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

@pll.Coordinator( PARALLEL )
def play(sess, agent, dataset, env, thread="worker_1", GAMES_PER_ITER=GAMES_PER_ITER):
    '''
    Plays games against OpenAI bot, sequentially or in parallel.

    * comments:

    IMPORTANT: if you're changing it, make sure the first three arguments are still sess, agent and dataset!
    In particular there's a depndency with the Worker.work function in parallelisation_tools
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
        if i_game % 1 == 0: #%20
            if agent.__class__ == logging_agent.Logging_Agent:
                #Currently disabling this due to stability reasons. See bug note
                #in the drive
                agent.playing_log(i_game, wins, losses)
            print("{0}. Game {1}. Agent: {2}, Opponent: {3}".format(thread, i_game, a_s, o_s))

agent = agents.wrapped_agent('main_agent') #note that: wrapped_agent is also called by parallelisation_tools
dataset = data.Dataset()

env = gym.make('Pong-v0')
logging.basicConfig(filename='info.log',level=logging.INFO)
logger = debugtools.Logger()

def main_function():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
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
            play(sess, agent, dataset, env, GAMES_PER_ITER) # A bit unhappy with GAMES_PER_ITER happening inside here, instead of
                                                            # in a foor loop, but have no better solution atm
            print("Size of dataset: {0}".format(dataset.size))
            print("Training:")
            for epoch in range(EPOCHS_PER_ITER):
                f,a,r = dataset.sample(TRAIN_BATCH_SIZE)
                loss = agent.train(sess,f,a,r)

                if epoch % 500 == 0:
                    if agent.__class__ == logging_agent.Logging_Agent:
                        agent.epoch_log(sess, epoch, loss)
                    print("Epoch {0}. Loss: {1}".format(epoch, loss))

            #Copy weights to workers to prepare for next run
            for worker in WORKERS:
                sess.run(worker.update_local_ops('main_agent',worker.name))
                if agent.__class__ == agents.BasicAgent:
                    #Ascertain that weights were copied correctly
                    pll.test_update(sess, worker, agent)
                    #yay


if __name__ == "__main__":
	main_function()
