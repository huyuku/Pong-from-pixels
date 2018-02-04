import numpy as np
import debugtools

class BasicAgent():

    def action(self, sess, diff_frame):
        return 0

    def gym_action(self, sess, diff_frame):
        return 3

    def train(self, sess, diff_frames, actions, rewards):
        return 0.5

    def set_time_start(self):
        return
