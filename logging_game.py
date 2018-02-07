import numpy as np
import dummy_agent
import gym
from config import *
import debugtools
import dummy_game

class Game():

	def __init__(self):
		self.wins = 0
		self.losses = 0
		self.games_played = 0
		self.logger = debugtools.Logger()

	def preprocess(self, frame):
		self.logger.set_time_start()
		ans = game.preprocess(frame)
		self.logger.logtime('Preprocessing', 1)
		return ans

	def create_play_data(self, session, agent, env, without_net=False, quick_play=False):
		f, a, r, o_s, a_s = game.create_play_data(session, agent, env, without_net, quick_play)
		self.games_played += 1
		if a_s>o_s:
			self.wins += 1
		else:
			self.losses += 1
		if self.games_played % 20 == 0:
			agent.playing_log(i_game, wins, losses)
			self.logger.loginfo("Game {0} score; Agent: {1}, Opponent: {2}".format(i_game, a_s, o_s))
			wins = 0
			losses = 0
		return f, a, r, o_s, a_s

def test():
	game_log = Game()
	print(game_log.preprocess(None))
	print(game_log.create_play_data(None, None, None))

if __name__ == "__main__":
	test()
