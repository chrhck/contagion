# -*- coding: utf-8 -*-

"""
Name: contagion.py
Authors: Christian Haack, Martina Karl, Stephan Meighen-Berger, Dominik Scholz
Main interface to the contagion module.
This package calculates the spread of an infection in
a given population. It allows for the introduction
of safety measures, such as social distancing and tracking.
"""

# Native modules
import sys
import logging
import pickle

# -----------------------------------------
# Package modules
from .config import config
from .infection import Infection
from .mc_sim import MC_Sim
from .measures import Measures
from .population import Population

# unless we put this class in __init__, __name__ will be contagion.contagion
_log = logging.getLogger('contagion')


class Contagion(object):
    """
    class: Contagion
    Interace to the contagion package. This class
    stores all methods required to run the simulation
    of the infection spread
    Parameters:
        -None
    Returns:
        -None
    """
    def __init__(self):
        """
        function: __init__
        Initializes the class Contagion.
        Here all run parameters are set.
        Parameters:
            -None
        Returns:
            -None
        """
        # Inputs
        self.__infected = config['infected']

        # Logger
        # creating file handler with debug messages
        fh = logging.FileHandler(config['log file handler'], mode='w')
        fh.setLevel(logging.DEBUG)
        # console logger with a higher log level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(config['debug level'])

        # Logging formatter
        fmt = '%(levelname)s: %(message)s'
        fmt_with_name = '[%(name)s] ' + fmt
        formatter_with_name = logging.Formatter(fmt=fmt_with_name)
        fh.setFormatter(formatter_with_name)
        # add class name to ch only when debugging
        if config['debug level'] == logging.DEBUG:
            ch.setFormatter(formatter_with_name)
        else:
            formatter = logging.Formatter(fmt=fmt)
            ch.setFormatter(formatter)

        _log.addHandler(fh)
        _log.addHandler(ch)
        _log.setLevel(logging.DEBUG)

        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Welcome to contagion!')
        _log.info('This package will help you model the spread of infections')
        _log.debug('Trying to catch some errors in the config')

        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        if config["re-use population"]:
            try:
                self.pop = pickle.load(
                    open(config["population storage"], "rb")
                )
                _log.debug('Population loaded')
            except ImportError:
                _log.error('Population file not found!')
                sys.exit('Population file not found!' +
                         ' Check the config file!')
        else:
            _log.info('Starting population construction')
            self.pop = Population().population
            # Storing for later
            _log.debug('Storing for later use')
            # pickle.dump(self.pop, open(config["population storage"],
            #                            "wb" ) )
        _log.info('Finished the population')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Starting the infection construction')
        self.infection = Infection()
        _log.info('Finished the infection construction')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Starting the measure construction')
        self.tracked = Measures().tracked
        _log.info('Finished the measure construction')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Setting the simulation framework')
        self.sim = self.__sim_realistic
        _log.info('Simulation framework set. Please type:')
        _log.info('self.sim(parameters) to run the simulation')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')

    @property
    def statistics(self):
        """
        function: statistics
        Getter functions for the simulation results
        from the simulation
        Parameters:
            -None
        Returns:
            -dic statistics:
                Stores the results from the simulation
        """
        return self.__mc_run.statistics

    @property
    def t(self):
        """
        function: t
        Getter functions for the used time array
        from the simulation
        Parameters:
            -None
        Returns:
            -np.array:
                The used time array
        """
        return self.__mc_run.time_array

    @property
    def R0(self):
        """
        function: R0
        Getter functions for the R0 value
        from the simulation
        Parameters:
            -None
        Returns:
            -float:
                The calculated R0
        """
        return self.__mc_run.R0

    def __sim_realistic(self):
        """
        function: __sim_realistic
        Realistic infection spread simulation
        Parameters:
            -None
        Returns:
            -np.array infected:
                The current population
        """
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        dt = config['time step']
        if dt > 1.:
            _log.error("Chosen time step too large!")
            sys.exit("Please run with time steps smaller than 1s!")
        _log.debug('Realistic run')
        self.__mc_run = MC_Sim(
            self.pop,
            self.infection,
            self.tracked
        )
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.info('Finished calculation')
        _log.info('The results are stored in a dictionary self.statistics')
        _log.info('Structure of dictionray:')
        _log.info(self.statistics.keys())
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        _log.debug('Dumping run settings into %s', config['config location'])
        with open(config['config location'], 'w') as f:
            for item in config.keys():
                print(item + ': ' + str(config[item]), file=f)
        _log.debug('Finished dump')
        _log.info('---------------------------------------------------')
        _log.info('---------------------------------------------------')
        # Closing log
        logging.shutdown()
