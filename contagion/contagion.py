# -*- coding: utf-8 -*-

"""
Name: contagion.py
Authors: Christian Haack, Martina Karl, Stephan Meighen-Berger
Main interface to the contagion module.
This package calculates the spread of an infection in
a given population. It allows for the introduction
of safety measures, such as social distancing and tracking.
"""

# Native modules
import sys
import logging
import numpy as np
from time import time
# -----------------------------------------
# Package modules
from .config import config as confi
from .infection import Infection
from .mc_sim import MC_Sim
from .measures import Measures
from .population import Population


class Contagion(object):
    """
    class: Contagion
    Interace to the contagion package. This class
    stores all methods required to run the simulation
    of the infection spread
    Parameters:
        -optional dic config:
            The dictionary from the config file
    Returns:
        -None
    """
    def __init__(self, config=confi):
        """
        function: __init__
        Initializes the class Contagion.
        Here all run parameters are set.
        Parameters:
            -optional dic config:
                The dictionary from the config file
        Returns:
            -None
        """
        # Inputs
        self.__config = config
        self.__infected = self.__config['infected']

        # Logger
        # creating file handler with debug messages
        fh = logging.FileHandler(self.__config['log file handler'], mode='w')
        fh.setLevel(logging.DEBUG)
        # console logger with a higher log level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(self.__config['debug level'])

        # Logging formatter
        fmt = '%(levelname)s: %(message)s'
        fmt_with_name = '[%(name)s] ' + fmt
        formatter_with_name = logging.Formatter(fmt=fmt_with_name)
        fh.setFormatter(formatter_with_name)
        # add class name to ch only when debugging
        if self.__config['debug level'] == logging.DEBUG:
            ch.setFormatter(formatter_with_name)
        else:
            formatter = logging.Formatter(fmt=fmt)
            ch.setFormatter(formatter)

        # Basic config for all loggers
        logging.basicConfig(handlers=[fh, ch])
        # Creating logger user_info
        self.__log = logging.getLogger(self.__class__.__name__)
        self.__log.setLevel(logging.DEBUG)

        self.__log.info('---------------------------------------------------')
        self.__log.info('---------------------------------------------------')
        self.__log.info('Welcome to contagion!')
        self.__log.info('This package will help you model the spread of infections')
        self.__log.debug('Trying to catch some errors in the config')
        if not(self.__config.keys() == confi.keys()):
            self.__log.error('Error in config!')
            exit('Please check your input')
        self.__log.info('---------------------------------------------------')
        self.__log.info('---------------------------------------------------')
        self.__log.info('Starting population construction')
        self.pop = Population(self.__log, self.__config).population
        self.__log.info('Finished the population')
        self.__log.info('---------------------------------------------------')
        self.__log.info('---------------------------------------------------')
        self.__log.info('Starting the infection construction')
        self.infection = Infection(self.__log, self.__config)
        self.__log.info('Finished the infection construction')
        self.__log.info('---------------------------------------------------')
        self.__log.info('---------------------------------------------------')
        self.__log.info('Starting the measure construction')
        self.tracked = Measures(self.__log, self.__config).tracked
        self.__log.info('Finished the measure construction')
        self.__log.info('---------------------------------------------------')
        self.__log.info('---------------------------------------------------')
        self.__log.info('---------------------------------------------------')
        self.__log.info('---------------------------------------------------')
        self.__log.info('Setting the simulation framework')
        self.sim = self.__sim_realistic
        self.__log.info('Simulation framework set. Please type:')
        self.__log.info('self.sim(parameters) to run the simulation')
        self.__log.info('---------------------------------------------------')
        self.__log.info('---------------------------------------------------')

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
        self.__log.info('---------------------------------------------------')
        self.__log.info('---------------------------------------------------')
        dt = self.__config['time step']
        if dt > 1.:
            self.__log.error("Chosen time step too large!")
            exit("Please run with time steps smaller than 1s!")
        self.__log.debug('Realistic run')
        self.__mc_run = MC_Sim(
            self.pop,
            self.infection,
            self.tracked,
            self.__log,
            self.__config
        )
        self.__log.info('Structure of dictionray:')
        self.__log.info('["t", "total", "encounter", "shear", "history"]')
        self.__log.debug('    t: The time array')
        self.__log.debug('    total: The total emissions at each point in time')
        self.__log.debug('    encounter: The encounter emissions at each point in time')
        self.__log.debug('    shear: The shear emissions at each point in time')
        self.__log.debug('    history: The population at every point in time')
        self.__log.info('---------------------------------------------------')
        self.__log.info('---------------------------------------------------')
        self.__log.info('Finished calculation')
        self.__log.debug('The results are stored in a dictionary self.statistics')
        self.__log.debug('Available keys are:')
        self.__log.debug('"contacts", "infections", "recovered", "immune", "infectious", "susceptible"')
        self.__log.info('---------------------------------------------------')
        self.__log.info('---------------------------------------------------')
        self.__log.debug('Dumping run settings into %s', self.__config['config location'])
        with open(self.__config['config location'], 'w') as f:
            for item in self.__config.keys():
                print(item + ': ' + str(self.__config[item]), file=f)
        self.__log.debug('Finished dump')
        self.__log.info('---------------------------------------------------')
        self.__log.info('---------------------------------------------------')
        # Closing log
        logging.shutdown()
