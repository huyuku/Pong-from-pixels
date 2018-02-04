#HYPERPARAMETERS AND SETTINGS
NUM_ITERATIONS = 100
GAMES_PER_ITER = 100
EPOCHS_PER_ITER = 50
TRAIN_BATCH_SIZE = 1000

LEARNING_RATE = 0.01
HIDDEN_SIZE = 100

WITHOUT_NET = False
QUICK_PLAY = bool(int(input("Quick self-play? 0 for no and 1 for yes\n"))) #For testing
debug = False
print_analytics = True
