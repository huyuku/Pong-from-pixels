import tensorflow as tf
import game
import gym

WORKERS = [] # just a nonsense list in order to prevent syntax errors, as the real WORKERS is in main.py

def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


class Coordinator():
    '''Wraps a function in order to parallelise it

    *arguments

    condition
        if False, calling the class as a decorator just returns the decorated function f
        if True, each worker in the global WORKER list is started in a separate thread on f

    *comments

    not sure if the sleep timer is necessary. It's only there because the Juliani functions
    this was inspired by uses it
    '''
    def __init__(self, condition):
        # We only want to use the coordinator if the algorithm is run in parallel
        self.condition = condition
        self.coord = tf.train.Coordinator()

    def __call__(self, f):
        if self.condition:
            print("I'm here!")
            def wrapper(*args):
                sess = args[0]
                worker_threads = []
                global WORKERS
                for worker in WORKERS:
                    worker.sess = sess
                    worker_work = lambda: worker.work(f)
                    t = threading.Thread(target=(worker_work))
                    t.start()
                    sleep(0.1)
                    worker_threads.append(t)
                self.coord.join(worker_threads)
            return wrapper
        return f

class Worker():
    '''Executes play function as a separate thread'''
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
        play_function(self.sess, self.dataset, self.local_agent, self.env, self.name)

num_workers = multiprocessing.cpu_count() # Set WORKERS to number of available CPU threads
WORKERS = []
# Create worker classes
num_gpus = 4
gpu_idx = 0
test_on_cpu = True
for i in range(num_workers):
    WORKERS.append(pll.Worker(str(i), wrapped_agent("worker_" + str(i)), gpu_idx, dataset, test_on_cpu=test_on_cpu))
    gpu_idx += 1
    if gpu_idx > num_gpus-1:
        gpu_idx = 0
