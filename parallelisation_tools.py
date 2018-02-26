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
    def __init__(self, parallelise):
        # We only want to use the coordinator if the algorithm is run in parallel
        self.parallelise = parallelise
        self.coord = tf.train.Coordinator()
        self.workers_have_been_created = False

    def __call__(self, f):
        if self.parallelise:
            def wrapper(*args):
                sess = args[0]
                Agent_class = args[1].__class__
                dataset = args[2]
                if not self.workers_have_been_created:
                    create_workers(Agent_class, dataset)
                    sess.run(tf.global_variables_initializer())
                    self.workers_have_been_created = True
                threads = []
                for worker in WORKERS:
                    worker.sess = sess
                    worker_work = lambda: worker.work(f)
                    t = threading.Thread(target=(worker_work))
                    t.start()
                    threads.append(t)
                self.coord.join(threads)
            return wrapper
        else:
            return f

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
    def __init__(self, Agent_class, thread, dataset, envname='Pong-v0'):
        self.name = "worker_" + str(thread)
        self.thread = thread
        self.dataset = dataset
        self.sess = None
        self.env = gym.make(envname)
        self.first_call = True

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
        play_function(self.sess, self.local_agent, self.dataset, self.env, self.name, GAMES_PER_ITER_PER_THREAD)

def create_workers(Agent_class, dataset):
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
        WORKERS.append(Worker(Agent_class, thread, dataset))
