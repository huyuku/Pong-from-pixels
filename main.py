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

def test(sess, main_agent, newly_trained_agent, dataset, env, thread="worker_1", GAMES_PER_ITER=GAMES_PER_ITER):
    '''
    Tests whether newly trained agent outperforms previous agent, and if so updates it and returns True

    * comments:

    The best way to do this is to choose a false positive rate alpha (e.g. 5%) and then update
    the win threshold based on statistics gathered over time to match the false positive rate.
    Currently this is not implemented.

    Test is not parallelised.

    '''
    champion_score = 0
    challenger_score = 0
    for i_game in range(NUM_TEST_GAMES):
        f, a, r, main_o_s, main_a_s = game.create_play_data(sess, main_agent, env, without_net=WITHOUT_NET, quick_play=QUICK_PLAY)
        f, a, r, new_o_s, new_a_s = game.create_play_data(sess, newly_trained_agent, env, without_net=WITHOUT_NET, quick_play=QUICK_PLAY)
        champion_score += main_a_s
        challenger_score += new_a_s
        print("Test game ", i_game)
        print("{0}. Game {1}. Old agent: {2}, Opponent: {3}".format(thread, i_game, main_a_s, main_o_s))
        print("{0}. Game {1}. New agent: {2}, Opponent: {3}".format(thread, i_game, new_a_s, new_o_s))

    if challenger_score > champion_score+3: # The 3 makes this somewhat noise robust, but in a lame and MVP way
        return True
    else:
        return False



agent = agents.wrapped_agent('main_agent') #note that: wrapped_agent is also called by parallelisation_tools
challenger_agent = agents.wrapped_agent('challenger') #note that: wrapped_agent is also called by parallelisation_tools

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
        #    if i == 0: #reset dataset initially. The condition exists because we don't reset in case new weight updates were futile in the challenger test (see below).
            dataset.reset()
            print("Iteration {0}".format(i+1))
            print("Playing games:")
            play(sess, agent, dataset, env, GAMES_PER_ITER) # A bit unhappy with GAMES_PER_ITER happening inside here, instead of
                                                            # in a foor loop, but have no better solution atm
            print("Size of dataset: {0}".format(dataset.size))
            print("Training:")
            for epoch in range(EPOCHS_PER_ITER):
                f,a,r = dataset.sample(TRAIN_BATCH_SIZE)
                loss = challenger_agent.train(sess,f,a,r)

                if epoch % 500 == 0:
                    if agent.__class__ == logging_agent.Logging_Agent:
                        agent.epoch_log(sess, epoch, loss)
                    print("Epoch {0}. Loss: {1}".format(epoch, loss))

            #Only update the main agent if the new weights lead to better performance
            if test(sess, agent, challenger_agent, dataset, env, GAMES_PER_ITER):
                print("Training yielded a new champion! Updating main agent weights...")
                sess.run(pll.update_target_graph(challenger_agent.name, agent.name))
                #Copy weights to workers to prepare for next run
                for worker in WORKERS:
                    sess.run(worker.update_local_ops(agent.name, worker.name))
                    if agent.__class__ == agents.BasicAgent:
                        #Ascertain that weights were copied correctly
                        pll.test_update(sess, worker, agent)
            #    dataset.reset()
            else:
                print("Training didn't improve performance (and possibly worsened it). Discarding weight udpates...")
                sess.run(pll.update_target_graph(agent.name, challenger_agent.name))



if __name__ == "__main__":
	main_function()
