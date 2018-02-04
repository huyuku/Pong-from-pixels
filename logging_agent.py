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
        self.logger.set_time_start()
        loss = self.agent.train(sess, diff_frames, actions, rewards)
        self.logger.logtime('Training', 1)
        return loss

    def set_time_start(self):
        self.logger.set_time_start()

    def playing_log(self, number_of_games, wins, losses):
        logger.loginfo("%s games, score: %s wins, %s losses" % (number_of_games, wins, losses))
        logger.logtime('Playing iteration')
        self.logger.set_time_start()

    def epoch_log(self, sess, epoch_number, loss):
        logger.loginfo("Epoch %s loss: %s" % (epoch_number, loss))
        logger.logtime('Epoch iteration')
        self.logger.set_time_start()
        self.log_matrices(sess)

    def log_matrices(self, sess):
        self.logger.log_matrix('W1', sess.run(self.agent.W1))
        self.logger.log_matrix('W2', sess.run(self.agent.W2))
