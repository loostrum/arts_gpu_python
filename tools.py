#!/usr/bin/env python3

from time import time
from contextlib import contextmanager

import numpy as np
from numba import cuda


@contextmanager
def timer(name, output=None):
    """
    Timer context manager
    :param name: Name of function to time
    :param output: list to append runtime to
    """
    start = time()
    yield
    end = time()
    runtime = end - start
    print("{} took {:.2f} ms".format(name, runtime*1000))
    if output is not None:
        output.append(runtime)


def tuner(func, threadmin, threadmax, nitem, nloop=1, return_full=False):
    # number of threads and blocks to try
    nthread = np.arange(threadmin, threadmax+1)
    nblock = np.ceil(nthread / nitem).astype(int)

    # run function for each setting and store runtime
    runtimes = []
    for block, thread in zip(nblock, nthread):
        this_runtime = []
        for loop in range(nloop):
            with timer('Tuner nblock={} nthread={}'.format(block, thread), this_runtime):
                func(block, thread)
        runtimes.append(np.median(this_runtime))
    # get settings for shortest runtime
    ind = np.argmin(runtimes)
    print("Optimal nblock={}, nthread={}".format(nblock[ind], nthread[ind]))
    if return_full:
        return nblock, nthread, np.array(runtimes)
    return nblock[ind], nthread[ind]
