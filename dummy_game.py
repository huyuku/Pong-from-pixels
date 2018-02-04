import random
import numpy as np
import dummy_agent
import gym
from config import *
import debugtools

def preprocess(frame):
	return np.zeros([80*80])

def create_play_data(session, agent, env, without_net=False, quick_play=False):
	return np.zeros([80*80]), np.mat([1]), np.mat([1]), 1, 0

def test():
	print(preprocess(np.zeros([80*80])))
	print(create_play_data(None, None, None))

if __name__ == "__main__":
	test()
