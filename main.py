import tensorflow as tf
import numpy as np
import threading
import multiprocessing
import gym
from time import sleep
import agents
import game
import data
import logging
import logging_agent
from game import *
from config import *
from worker import *

PARALLEL = True


def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


class Coordinator:
    def __init__(self, condition):
        # We only want to use the coordinator if the algorithm is run in parallel
        self.condition = condition
        self.coord = tf.train.Coordinator()

    def __call__(self, f):
        if self.condition:
            def wrapper(*args):
                sess = args[0]
                worker_threads = []
                for worker in workers:
                    worker.sess = sess
                    worker_work = lambda: worker.work(f)
                    t = threading.Thread(target=(worker_work))
                    t.start()
                    sleep(0.5)
                    worker_threads.append(t)
                self.coord.join(worker_threads)
            return wrapper
        return f

@Coordinator( PARALLEL )
def play(sess, dataset, agent, env):
    '''Plays games against itself, sequentially or in parallel.
    IMPORTANT: if you're changing it, make sure the first argument is still sess!'''
    f, a, r, o_s, a_s = game.create_play_data(sess, agent, env, without_net=WITHOUT_NET, quick_play=QUICK_PLAY)
    dataset.add(f,a,r)

class Worker():
    def __init__(self, name, agent, gpu_idx, dataset, test_on_cpu=False):
        self.name = "worker_" + str(name)
        self.train_data = game.train_set()
        self.gpu_idx = str(gpu_idx)
        self.dataset = dataset
        self.sess = None
        self.env = env = gym.make('Pong-v0')
        if test_on_cpu:
            self.device = '/cpu:'
        else:
            self.device = '/gpu:'

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        with tf.device(self.device+self.gpu_idx):
            with tf.name_scope(name) as scope:
                self.local_agent = agent
                self.update_local_ops = update_target_graph('main_agent',self.name)

    def work(self, play_function):
        print("Starting play with worker "+self.name+" on GPU "+self.gpu_idx+"...")
        play_function(self.sess, self.dataset, self.local_agent, self.env)

asynchronous = True

agent = agents.BasicAgent('main_agent', HIDDEN_SIZE, LEARNING_RATE)
agent = logging_agent.Logging_Agent(agent)
dataset = data.Dataset()
if PARALLEL:
    num_workers = multiprocessing.cpu_count() # Set workers to number of available CPU threads
    workers = []
    # Create worker classes
    num_gpus = 4
    gpu_idx = 0
    test_on_cpu = True
    for i in range(num_workers):
        workers.append(Worker(str(i), agents.BasicAgent("worker_" + str(i), HIDDEN_SIZE, LEARNING_RATE), gpu_idx, dataset, test_on_cpu=test_on_cpu))
        gpu_idx += 1
        if gpu_idx > num_gpus-1:
            gpu_idx = 0


env = gym.make('Pong-v0')
logging.basicConfig(filename='info.log',level=logging.INFO)
logger = debugtools.Logger()

def main_function():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        wins = 0
        losses = 0
        agent.set_time_start()
        for i in range(NUM_ITERATIONS):
            dataset.reset()
            print("Iteration {0}".format(i+1))
            print("Playing games:")
            for i_game in range(GAMES_PER_ITER):
                play(sess, dataset, agent, env)
#                f, a, r, o_s, a_s = game.create_play_data(sess, agent, env, without_net=WITHOUT_NET, quick_play=QUICK_PLAY)
#                dataset.add(f,a,r)
#                if a_s>o_s:
#                    wins += 1
#                else:
#                    losses += 1
#                if i_game % 20 == 0:
#                    agent.playing_log(i_game, wins, losses)
                    #print("Game {0}. Agent: {1}, Opponent: {2}".format(i_game, a_s, o_s))
            print("Size of dataset: {0}".format(dataset.size))
            print("Training:")
            for epoch in range(EPOCHS_PER_ITER):

                f,a,r = dataset.sample(TRAIN_BATCH_SIZE)
                loss = agent.train(sess,f,a,r)
                for worker in workers:
                    worker.update_local_ops
                if epoch % 500 == 0:
                    agent.epoch_log(sess, epoch, loss)
                    #print("Epoch {0}. Loss: {1}".format(epoch, loss))


if __name__ == "__main__":
    main_function()
