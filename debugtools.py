from config import *
import time
import logging
import numpy as np
#import tensorflow as tf

class Logger():

    def __init__(self):
        self.t1 = time.time()
        self.t2 = 0

    def set_time_start(self):
        self.t1 = time.time()
        self.t2 = 0

    def logtime(self, process_name, limit=0):
        self.t2 = time.time()
        if (self.t2-self.t1)>limit:
            logging.info(process_name + " took %s seconds to complete." % (self.t2-self.t1))
            if print_analytics:
                print(process_name + " took %s seconds to complete." % (self.t2-self.t1))
        self.t1 = self.t2

    def loginfo(self, string):
        logging.info(string)
        if print_analytics:
            print(string)

    def log_matrix(self, name, matrix):
        #tf.Print(matrix)
        self.loginfo('Matrix '+name+':')
        self.loginfo('')
        self.loginfo(np.array_str(matrix))
        self.loginfo('')
        if print_analytics:
            print('Matrix '+name+':\n')
            print(np.array_str(matrix)+'\n')

    #def print_matrix(self, matrix, mat_height, mat_width)
        #fig = plt.figure()
        #ax = fig2.add_subplot(mat_height, mat_width)
        #ax.set_xticks(())
        #ax.set_yticks(())
        #ax.imshow(matrix.reshape(mat_height, mat_width), cmap='Greys_r')
