import tensorflow as tf
import game
import data
import gym
from config import NUM_CPUS, NUM_GPUS, TEST_ON_CPU, WORKERS
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
    '''Wraps a function in order to parallelise it, and calls the a function to create the separate workers

    * arguments to __init__:

    condition
        if False, calling the class as a decorator just returns the decorated function f
        if True, each worker in the global WORKER list is started in a separate thread on f

    * arguments to __call__:

    f
        the function to be wrapped
    args
        hmmm what are they doing there?

    * comments:

    not sure if the sleep timer is necessary. It's only there because the Juliani functions
    this was inspired by uses it
    '''
    def __init__(self, condition):
        # We only want to use the coordinator if the algorithm is run in parallel
        self.condition = condition
        self.coord = tf.train.Coordinator()
        self.workers_have_been_created = False

    def __call__(self, f):
        if self.condition:
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
                    sleep(0.1)
                    threads.append(t)
                self.coord.join(threads)
            return wrapper
        return f

class Worker():
    '''Executes play function as a separate thread'''
    def __init__(self, name, Agent_class, gpu_idx, dataset):
        self.name = "worker_" + str(name)
        self.train_data = data.Dataset()
        self.gpu_idx = str(gpu_idx)
        self.dataset = dataset
        self.sess = None
        self.env = env = gym.make('Pong-v0')
        self.first_call = True
        if TEST_ON_CPU:
            self.device = '/cpu:'
        else:
            self.device = '/device:GPU:'

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        with tf.device(self.device+self.gpu_idx):
            with tf.name_scope(name) as scope:
                self.local_agent = Agent_class(self.name)
                #should't need to pass the function here, should be able to include arguments
                self.update_local_ops = update_target_graph

    def work(self, play_function):
        if self.first_call:
            print("Starting play with worker "+self.name+" on GPU "+self.gpu_idx+"...")
            self.first_call = False
        play_function(self.sess, self.local_agent, self.dataset, self.env, self.name)

def create_workers(Agent_class, dataset):
    '''
    Initialises instances of the Worker class for the desired number of parallel threads to run

    * arguments:

    Agent_class
        the class of the agent to be used as a worker on the different threads
    dataset
        where to store the work conducted by the workers
    '''
    num_workers = NUM_CPUS/4
    global WORKERS
    gpu_idx = 0
    for i in range(num_workers):
        WORKERS.append(Worker(str(i), Agent_class, gpu_idx, dataset))
        gpu_idx += 1
        if gpu_idx > NUM_GPUS-1:
            gpu_idx = 0
