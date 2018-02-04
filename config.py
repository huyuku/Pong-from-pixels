#HYPERPARAMETERS AND SETTINGS
num_self_play_games = 100
num_train_epochs = 50 #Change to something plausible later
train_batch_size = 100
NUM_ITERATIONS = 100
learning_rate = 0.01
hidden_size = 100
WITHOUT_NET = False
quick_self_play = bool(int(input("Quick self-play? 0 for no and 1 for yes\n"))) #For testing
debug = False
print_analytics = True
