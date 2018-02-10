from tensorflow.python.client import device_lib
import multiprocessing

#HYPERPARAMETERS AND SETTINGS
NUM_ITERATIONS = 100
GAMES_PER_ITER = 1
EPOCHS_PER_ITER = 10
TRAIN_BATCH_SIZE = 100

LEARNING_RATE = 0.001
HIDDEN_SIZE = 200

#ANALYTICS AND DEBUGGING
WITHOUT_NET = False
QUICK_PLAY = False
RENDER = False
debug = False
print_analytics = True

#PARALLELISATION
#Devices available
NUM_CPUS = multiprocessing.cpu_count() # Set workers to number of available CPU threads
GPUs = [x.name for x in device_lib.list_local_devices() if x.device_type == 'GPU']
NUM_GPUS = len(GPUs)

#Main setting
PARALLEL  = True
if PARALLEL:
    print("Detected %s CPUs and %s  GPUs" % (NUM_CPUS, NUM_GPUS))

#Settings for local debuging on laptop
if NUM_GPUS > 0: TEST_ON_CPU = False
else: TEST_ON_CPU = PARALLEL
if TEST_ON_CPU: NUM_GPUS = 4
if TEST_ON_CPU:
    print("But simulating exeuction on 4 GPUs for testing purposes")
