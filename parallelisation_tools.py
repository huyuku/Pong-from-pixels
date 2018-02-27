import tensorflow as tf
import game
import data
import gym
from config import NUM_CPUS, NUM_GPUS, NUM_THREADS, USE_GPU, TEST_GPU_ON_CPU, GAMES_PER_ITER_PER_THREAD, WORKERS
import threading
import multiprocessing
from time import sleep
import numpy as np

def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def test_update(sess, worker, agent):
    '''
    Tests that the variable called W1 was succesfully copied between the main agent
    and its worker clone, as a proxy for all variables being copied correctly.
    This test is currently only compatible with agents
    that have variables called "W1".
    '''
    sess.run(worker.update_local_ops('main_agent',worker.name))
    w1, w2 = sess.run([worker.local_agent.W1, agent.W1])
    np.testing.assert_array_equal(w1,w2)

class Coordinator():
    '''
    Wraps a data generating function it's called on to add parallel threads.

    Creates NUM_THREADS workers spread across available devices.

    * arguments to __init__:

    parallelise
        if False, calling the class as a decorator on f just returns f
        otherwise, does all of the parallelising as described above.

    * arguments to __call__:

    f
        the function to be wrapped

    '''
    def __init__(self):
        self.coord = tf.train.Coordinator()

    def setup(self, sess, Agent_class, envname, dataset, scorekeeper):
        create_workers(sess, Agent_class, envname, dataset, scorekeeper)
        sess.run(tf.global_variables_initializer())

    def __call__(self, f):
        def wrapper(**kwargs):
            sess = kwargs['sess']
            dataset = kwargs['dataset']
            scorekeeper = kwargs['scorekeeper']
            threads = []
            for worker in WORKERS:
                worker_work = lambda: worker.work(f)
                t = threading.Thread(target=(worker_work))
                t.start()
                threads.append(t)
            self.coord.join(threads)
        return wrapper

class Worker():
    '''
    Encapsultes a thread with a copy of the environment and an agent.

    * constructor arguments:
        Agent_class - class of agent to be instantiated
        thread      - thread number
        dataset     - data.Dataset to store the created train data in
        envname     - the environment name fed into gym.make(envname)

    * attributes:
        name        - string
        thread      - int
        sess        - tf.Session object
        env         - gym.Env object

    * methods:
        work(play_function)
            calls play_function on itself.
    '''
    def __init__(self, sess, Agent_class, thread, dataset, scorekeeper, envname='Pong-v0'):
        self.name = "worker_" + str(thread)
        self.sess = sess
        self.thread = thread
        self.env = gym.make(envname)
        self.first_call = True
        self.dataset = dataset
        self.scorekeeper = scorekeeper

        if USE_GPU:
            self.device = '/device:GPU:'
            self.device_id = str(self.thread % NUM_GPUS)
        else:
            self.device = '/cpu:'
            self.device_id = str(self.thread % NUM_CPUS)

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        with tf.device(self.device+self.device_id):
            with tf.name_scope(self.name) as scope:
                self.local_agent = Agent_class(self.name)
                #should't need to pass the function here, should be able to include arguments
                self.update_local_ops = update_target_graph

    def work(self, play_function):
        if self.first_call:
            print("Starting play with {} on device {}...".format(self.name, self.device_id))
            self.first_call = False
        play_function(sess=self.sess,
                      dataset=self.dataset,
                      scorekeeper=self.scorekeeper,
                      env=self.env,
                      agent=self.local_agent,
                      thread=self.name,
                      games_per_iter=GAMES_PER_ITER_PER_THREAD)

def create_workers(sess, Agent_class, envname, dataset, scorekeeper):
    '''
    Initialises instances of the Worker class for the desired number of parallel threads to run

    * arguments:

    Agent_class
        the class of the agent to be used as a worker on the different threads
    dataset
        where to store the work conducted by the workers
    '''
    global WORKERS
    for thread in range(NUM_THREADS):
        print("Creating thread {}...".format(thread), end='\r')
        WORKERS.append(Worker(sess, Agent_class, thread, dataset, scorekeeper, envname))
    print()

def update_workers(sess, source_agent):
    global WORKERS
    for worker, i in zip(WORKERS, range(len(WORKERS))):
        print("Updating worker {}/{}...".format(i+1, len(WORKERS)), end="\r")
        sess.run(worker.update_local_ops(source_agent.name, worker.name))
    print()
