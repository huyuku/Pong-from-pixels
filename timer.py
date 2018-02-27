"""
An ultra-simple timing module for easy profiling.
"""

import time

__last_time__ = 0

def click():
    """
    returns the time in seconds elapsed since last click.

    if not previously clicked, returns number of seconds since the Epoch.
    """
    global __last_time__
    tmp = __last_time__
    __last_time__ = time.time()
    return __last_time__ - tmp
