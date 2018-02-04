import tensorflow as tf
import numpy as np
import gym
import agents
import game
import data
import logging
import logging_agent
from game import *
from config import *

agent = agents.BasicAgent(HIDDEN_SIZE, LEARNING_RATE)
agent = logging_agent.Logging_Agent(agent)
game_object = logging_game.Game()
env = gym.make('Pong-v0')
logging.basicConfig(filename='info.log',level=logging.INFO)
logger = debugtools.Logger()

def main_function():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        dataset = data.Dataset()
        wins = 0
        losses = 0
        scored = 0
        agent.set_time_start()
        for i in range(NUM_ITERATIONS):
            dataset.reset()
            print("Iteration {0}".format(i+1))
            print("Playing games...")
            for i_game in range(GAMES_PER_ITER):
                f, a, r, o_s, a_s = game_object.create_play_data(sess, agent, env, without_net=WITHOUT_NET, quick_play=QUICK_PLAY)
                dataset.add(f,a,r)
<<<<<<< HEAD
                #if a_s>o_s:
                #    wins += 1
                #else:
                #    losses += 1
                #if i_game % 20 == 0:
                #    agent.playing_log(i_game, wins, losses)
                #    print("Game {0} score; Agent: {1}, Opponent: {2}".format(i_game, a_s, o_s))
                #    wins = 0
                #    losses = 0
=======
                scored += a_s
                if a_s>o_s:
                    wins += 1
                else:
                    losses += 1
                if i_game % 20 == 0:
                    agent.playing_log(i_game, wins, losses)
                print(" {0} games played. Wins/Losses: {1}/{2}. Goals scored: {3}".format(i_game+1, wins, losses, scored), end='\r')
            print()
>>>>>>> 60b1e5fb66c8af09c02f8569700a719638ecabac
            print("Size of dataset: {0}".format(dataset.size))
            print("Training...")
            for epoch in range(EPOCHS_PER_ITER):
                f,a,r = dataset.sample(TRAIN_BATCH_SIZE)
                loss = agent.train(sess,f,a,r)
                print(" {0} epochs trained. Loss on last epoch: {1:.2f}".format(epoch+1, loss), end='\r')
            print()


if __name__ == "__main__":
    main_function()
