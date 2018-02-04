from config import *
import time
import logging

class Logger():

    def __init__(self):
        self.t1 = time.time()
        self.t2 = 0

    def settimestart():
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
