from config import *
import debugtools

class Logging_Agent():

    def __init__(self, agent):
        self.agent = agent
        self.logger = debugtools.Logger()
        self.epochlogger = debugtools.Logger()
        self.taken_train_steps = 0 #not used atm

    def action(self, sess, diff_frame):
        #self.logger.set_time_start()
        ans = self.agent.action(sess, diff_frame)
        #self.logger.logtime('Action generation', 1)
        return ans

    def gym_action(self, sess, diff_frame):
        return self.agent.gym_action(sess, diff_frame)

    def train(self, sess, diff_frames, actions, rewards):
        #self.logger.set_time_start()
        loss = self.agent.train(sess, diff_frames, actions, rewards)
        #self.logger.logtime('Training', 1)
        return loss

    def set_time_start(self):
        self.logger.set_time_start()

    def playing_log(self, number_of_games, wins, losses):
        self.logger.loginfo("%s games, score: %s wins, %s losses" % (number_of_games, wins, losses))
        self.logger.logtime('Playing iteration')
        self.logger.set_time_start()

    def epoch_log(self, sess, epoch_number, loss):
        self.epochlogger.loginfo("Epoch %s loss: %s" % (epoch_number, loss))
        self.epochlogger.logtime('Epoch iteration %s' % epoch_number)
        #self.logger.set_time_start()
        #self.log_matrices(sess)

    def log_matrices(self, sess):
        self.logger.log_matrix('W1', sess.run(self.agent.W1))
        self.logger.log_matrix('W2', sess.run(self.agent.W2))
