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
import timer
from game import *
from config import *
from worker import *

coordinator = pll.Coordinator()
@coordinator
def play(sess, dataset, scorekeeper,
         env=None,
         agent=None,
         thread="main",
         games_per_iter=GAMES_PER_ITER):
    '''
    Plays games against OpenAI bot in parallel.

    The code in this function runs in each thread independently.
    '''
    for i_game in range(games_per_iter):
        f, a, r, o_s, a_s = game.create_play_data(sess, agent, env, without_net=WITHOUT_NET, quick_play=QUICK_PLAY)
        dataset.add(f,a,r)
        scorekeeper.add_game_results(a_s, o_s)
        if i_game % 1 == 0: #%20
            if agent.__class__ == logging_agent.Logging_Agent:
                #Currently disabling this due to stability reasons. See bug note
                #in the drive
                agent.playing_log(i_game, wins, losses)
            print("{0}. Game {1}/{2}. Agent: {3}, Opponent: {4}".format(thread, i_game+1, games_per_iter, a_s, o_s))

dataset = data.Dataset()
scorekeeper = data.Scorekeeper()

logging.basicConfig(filename='info.log',level=logging.INFO)

current_agent = agents.ConvNetAgent('current_agent')
saver = tf.train.Saver(keep_checkpoint_every_n_hours=1,
                       save_relative_paths=True)

backup_agent = agents.ConvNetAgent('backup_agent')

def main_function():
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        coordinator.setup(sess, agents.ConvNetAgent, 'Pong-v0', dataset, scorekeeper)

        for i in range(NUM_ITERATIONS):
            dataset.reset()
            scorekeeper.reset()
            print("Iteration {}".format(i+1))
            print("Updating workers...")
            pll.update_workers(sess, current_agent)
            print("workers up to date.")
            print("Playing games:")
            timer.click()
            play(sess=sess, dataset=dataset, scorekeeper=scorekeeper)
            print("playing took: {} seconds".format(timer.click()))
            print("Size of dataset: {}".format(dataset.size))
            scorekeeper.compute_statistics()
            print("average score for this iteration is {}".format(scorekeeper.current_avg))
            print("previous average score was {}".format(scorekeeper.previous_avg))
            if scorekeeper.current_avg > scorekeeper.previous_avg:
                print("new agent is better. updating backup...")
                sess.run(pll.update_target_graph(current_agent.name, backup_agent.name))
                scorekeeper.update_previous()
                print("backup up to date.")
                print("Training current agent:")
                for epoch in range(EPOCHS_PER_ITER):
                    f,a,r = dataset.sample(TRAIN_BATCH_SIZE, RANDOM_SAMPLES)
                    loss = current_agent.train(sess,f,a,r)
                    print("Epoch {0}. Loss: {1}".format(epoch, loss), end='\r')
                print()
            else:
                print("new agent is worse. loading from backup...")
                sess.run(pll.update_target_graph(backup_agent.name, current_agent.name))
                print("backup loaded.")

            if i % 10 == 0:
                print("Saving model...")
                saver.save(sess, 'models/my_model', global_step=i)



if __name__ == "__main__":
	main_function()
