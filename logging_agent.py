from config import *
import debugtools

class Logging_Agent():

    def __init__(self, agent):
        self.agent = agent
        self.logger = debugtools.Logger()

    def action(self, sess, diff_frame):
        #self.logger.set_time_start()
        ans = self.agent.action(sess, diff_frame)
        #self.logger.logtime('Action generation', 1)
        return ans

    def gym_action(self, sess, diff_frame):
        return self.agent.gym_action(sess, diff_frame)

    def train(self, sess, diff_frames, actions, rewards):
        self.timer.setstart()
        loss = self.agent.train(sess, diff_frames, actions, rewards)
        self.timer.logtime('Training', 1)
        return loss

    def log_matrices(sess):
        self.logger.log_matrix('W1', sess.run(self.agent.W1))
        self.logger.log_matrix('W2', sess.run(self.agent.W2))
