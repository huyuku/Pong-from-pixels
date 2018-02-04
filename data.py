import numpy as np

class Dataset():
	'''
	This class stores data from past games played by the agent.

	* functions:

	add(diff_frames, actions, rewards)
		adds the rewards, actions, and diff_frames arrays into the dataset.

	reset()
		removes all recorded data.

	sample(batch_size, random=False)
		returns a tuple (rewards, actions, diff_frames), where each
		element has height batch_size.

		if random=False, iterates over the data in order.
		if random=True, samples the data randomly.
	'''
	def __init__(self):
		self.diff_frames = None
		self.actions = None
		self.rewards = None
		self.size = 0
		self.pos = 0

	def add(self, diff_frames, actions, rewards):
		#validate input
		h_frames, _= diff_frames.shape
		h_actions = actions.size
		h_rewards = rewards.size
		assert h_frames == h_actions == h_rewards, "arguments to Dataset.add() need to have equal height"

		#add data into dataset
		if self.size==0:
			self.diff_frames = diff_frames
			self.actions = actions
			self.rewards = rewards
		else:
			self.diff_frames = np.stack((self.diff_frames, diff_frames), axis=0)
			self.actions     = np.concatenate((self.actions, actions))
			self.rewards     = np.concatenate((self.rewards, rewards))
		self.size = self.size + h_frames

	def reset(self):
		self.diff_frames = None
		self.actions = None
		self.rewards = None
		self.size = 0
		self.pos = 0

	def sample(self, batch_size, random=False):
		assert self.size != 0, "dataset must not be empty"
		assert batch_size <= self.size, "batch must be smaller than full dataset"
		if not random:
			indices = np.arange(start=self.pos,
								stop=self.pos+batch_size)
			indices = np.mod(indices, self.size) #cycle to start if overflows
			self.pos = (self.pos + batch_size) % self.size
		else:
			indices = np.random.permutation(np.arange(self.size))[:batch_size]
		return self.diff_frames[indices], self.actions[indices], self.rewards[indices]


#some tests
def overflow_test():
	data = Dataset()

	rewards = np.arange(1000)
	actions = np.arange(1000,0,-1)
	frames  = np.ones((1000,100))
	data.add(frames, actions, rewards)

	data.pos = 995
	_, _, t = data.sample(10)
	if (t == np.mod(np.arange(995, 1005), 1000)).all():
		print("overflow test passed")
	else:
		print("overflow test failed. Output:")
		print(t)
		print("expected:")
		print(np.mod(np.arange(995, 1005), 1000))

def inequality_test():
	data = Dataset()

	rewards = np.arange(999)
	actions = np.arange(1000,0,-1)
	frames  = np.ones((1000,100))
	try:
		data.add(frames, actions, rewards)
	except Exception as e:
		print("input inequality test passed")

def progression_test():
	data = Dataset()

	rewards = np.arange(1000)
	actions = np.arange(1000,0,-1)
	frames  = np.ones((1000,100))
	data.add(frames, actions, rewards)

	_, _, t1 = data.sample(1)
	_, _, t2 = data.sample(1)
	if t2[0] == t1[0]+1:
		print("progression test passed")
	else:
		print("progression test failed:")
		print("t1: ", t1)
		print("t2: ", t2)
