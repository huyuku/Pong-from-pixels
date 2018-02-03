import numpy as np

class Training_Data():
	'''
	This class stores data from past games played by the agent.

	* functions:

	add(rewards, actions, diff_frames)
		adds the rewards, actions, and frames arrays provided into the dataset.

    reset()
        removes all recorded data.

	sample(batch_size, random=False)
		returns a tuple (rewards, actions, diff_frames), where each
        element has height batch_size.

        if random=False, iterates over the data in order.
        if random=True, samples the data randomly.
	'''

	def add(self, diff_frames, actions, rewards):
        pass

	def sample(self,batch_size):
        pass
