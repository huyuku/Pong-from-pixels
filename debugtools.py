from config import *
import time
import logging
#import tensorflow as tf

class Logger():

    def __init__(self):
        self.t1 = time.time()
        self.t2 = 0

    def set_time_start():
        self.t1 = time.time()
        self.t2 = 0

    def logtime(process_name, limit=0):
        self.t2 = time.time()
        if (t2-t1)>limit:
            logging.info(process_name + " took %s seconds to complete." % (t2-t1))
            if print_analytics:
                print(process_name + " took %s seconds to complete." % (t2-t1))
        self.t1 = t2

    def loginfo(string):
        logging.info(string)
        if print_analytics:
            print(string)

    def log_matrix(name, matrix):
        #tf.Print(matrix)
        loginfo('Matrix '+name+ ':')
        loginfo('')
        loginfo(numpy.array_str(matrix))
        loginfo('')

    #def print_matrix(matrix, mat_height, mat_width)
        #fig = plt.figure()
        #ax = fig2.add_subplot(mat_height, mat_width)
        #ax.set_xticks(())
        #ax.set_yticks(())
        #ax.imshow(matrix.reshape(mat_height, mat_width), cmap='Greys_r')
