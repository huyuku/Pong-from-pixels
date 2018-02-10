import tensorflow as tf
import game
import data
import gym
from config import NUM_CPUS, NUM_GPUS, TEST_ON_CPU
import threading
import multiprocessing
from time import sleep

def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


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

    def __call__(self, f):
        if self.condition:
            def wrapper(*args):
                sess = args[0]
                Agent_class = args[1].__class__
                dataset = args[2]
                workers = create_workers(Agent_class, dataset)
                threads = []
                for worker in workers:
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
    def __init__(self, name, agent, gpu_idx, dataset, test_on_cpu=False):
        self.name = "worker_" + str(name)
        self.train_data = data.Dataset()
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
    num_workers = NUM_CPUS
    workers = []
    gpu_idx = 0
    for i in range(num_workers):
        workers.append(Worker(str(i), Agent_class("worker_" + str(i)), gpu_idx, dataset, test_on_cpu=TEST_ON_CPU))
        gpu_idx += 1
        if gpu_idx > NUM_GPUS-1:
            gpu_idx = 0

    return workers
