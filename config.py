from tensorflow.python.client import device_lib
import multiprocessing

#HYPERPARAMETERS AND SETTINGS
NUM_ITERATIONS = 10
GAMES_PER_ITER = 16
EPOCHS_PER_ITER = 1
TRAIN_BATCH_SIZE = 1000
RANDOM_SAMPLES = True

LEARNING_RATE = 0.0001
HIDDEN_SIZE = 200

#ANALYTICS AND DEBUGGING
WITHOUT_NET = False
QUICK_PLAY = False
RENDER = False
debug = False
print_analytics = True

#PARALLELISATION


#Main setting
PARALLEL = True
NUM_THREADS = 8
USE_GPU = False
TEST_GPU_ON_CPU = False

#Devices available
NUM_CPUS = multiprocessing.cpu_count() # Set workers to number of available CPU threads
GPUs = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
NUM_GPUS = len(GPUs)
if PARALLEL:
    print("Detected %s CPUs and %s  GPUs" % (NUM_CPUS, NUM_GPUS))
assert NUM_THREADS <= NUM_CPUS, "NUM_THREADS is higher than what is supported by the CPU"

#Some extra configs generated from the above
GAMES_PER_ITER_PER_THREAD = int(GAMES_PER_ITER / NUM_THREADS)
if not PARALLEL:
    NUM_THREADS = 1

#Settings for local debuging on laptop

if TEST_GPU_ON_CPU:
    NUM_CPUS = 4
    print("But simulating execution on {} GPUs for testing purposes".format(NUM_CPUS))

WORKERS = []
