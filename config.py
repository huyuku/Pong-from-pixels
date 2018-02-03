#HYPERPARAMETERS AND SETTINGS
num_self_play_games = 1000
num_train_epochs = 100 #Change to something plausible later
train_batch_size = 100
num_iterations = 10
learning_rate = 0.01
hidden_size = 100
without_net = False
quick_self_play = bool(int(input("Quick self-play? 0 for no and 1 for yes"))) #For testing
debug = False
print_analytics = True
