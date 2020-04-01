#! /usr/local/bin/python3

import sys
import os
import time
import cProfile
import pstats
try:
    from pstats import SortKey
except ImportError:
    # SortKey is python > 3.6
    SortKey = None

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), '../'))

from contagion import Contagion, config


def speed(c):
    # measure execution time of simulation
    print('Starting speed test')
    start = time.time()
    c.sim()
    end = time.time()

    print('Finished speed test')
    print(end-start)


def profiling(c):
    # profiling of simulation
    local = {'c': c}
    print('Starting profiling')
    cProfile.runctx('c.sim()', {}, local, filename='stats')
    print('Finished profiling')

    p = pstats.Stats('stats')
    if SortKey:
        p.strip_dirs().sort_stats(SortKey.TIME).print_stats()
    else:
        p.strip_dirs().print_stats()


def setup():
    # config paths are optimized for examples/*.ipynb, but crash when used
    # from script
    config['log file handler'] = 'contagion.log'
    config['config location'] = 'config.txt'

    print('Creating simulation')
    c = Contagion()
    return c


def main():
    c = setup()
    speed(c)
    # profiling(c)


if __name__ == '__main__':
    main()
