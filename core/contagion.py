"""
Name: contagion.py
Authors: Stephan Meighen-Berger
Main interface to the contagion module.
This package calculates the spread of an infection in
a given population. It allows for the introduction
of safety measures, such as social distancing and tracking.
"""

"Imports"
# Native modules
import logging
import numpy as np
from time import time
# -----------------------------------------
# Package modules
from con_config import config as confi
# -----------------------------------------
# Realistic sim imports
from con_population import CON_population
# -----------------------------------------
# The random walk imports
from con_adamah import con_adamah
from con_random_walk import CON_random_walk
from con_infection import CON_infection
from con_mc_sim import CON_mc_sim
from con_measures import CON_measures

class CONTAGION(object):
    """
    class: CONTAGION
    Interace to the contagion package. This class
    stores all methods required to run the simulation
    of the infection spread
    Parameters:
        -int infected:
                The number of starting infections
        -optional dic config:
            The dictionary from the config file
    Returns:
        -None
    """
    def __init__(self, infected, config=confi):
        """
        function: __init__
        Initializes the class CONTAGION.
        Here all run parameters are set.
        Parameters:
            -int infected:
                The number of starting infections
            -optional dic config:
                The dictionary from the config file
        Returns:
            -None
        """
        # Inputs
        self.infected = infected
        self.config = config
        pop = self.config['population size']
        "Logger"
        # Basic config empty for now
        logging.basicConfig()
        # Creating logger user_info
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        self.log.propagate = False
        # creating file handler with debug messages
        self.fh = logging.FileHandler('../contagion.log', mode='w')
        self.fh.setLevel(logging.DEBUG)
        # console logger with a higher log level
        self.ch = logging.StreamHandler()
        self.ch.setLevel(self.config['debug level'])
        # Logging formatter
        formatter = logging.Formatter(
            fmt='%(levelname)s: %(message)s'
        )
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)
        # Adding the handlers
        self.log.addHandler(self.fh)
        self.log.addHandler(self.ch)
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Welcome to contagion!')
        self.log.info('This package will help you model the spread of infections')
        # Checking the type of the simulation
        self.log.info('Simulation type is set to ' + self.config['simulation type'])
        if self.config['simulation type'] == 'realistic':
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            self.log.info('Starting population construction')
            self.pop = CON_population(pop, self.log, self.config).population
            self.log.info('Finished the population')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            self.log.info('Starting the infection construction')
            self.infection = CON_infection(self.log, self.config)
            self.log.info('Finished the infection construction')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            self.log.info('Starting the measure construction')
            self.tracked = CON_measures(self.log, self.config).tracked
            self.log.info('Finished the measure construction')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            self.log.info('Setting the simulation framework')
            self.sim = self.__sim_realistic
            self.log.info('Simulation framework set. Please type:')
            self.log.info('self.sim(parameters) to run the simulation')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
        elif self.config['simulation type'] == 'random walk':
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            self.log.info('Creating the world')
            self.world = con_adamah(self.log, self.config)
            self.log.info('Finished world building')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            self.log.info('Setting the simulation framework')
            self.sim = self.__sim_random_walk
            self.log.info('Simulation framework set. Please type:')
            self.log.info('self.sim(parameters) to run the simulation')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
        else:
            self.log.error('Simulation type unknown: ' + self.config['simulation type'])
            exit('Please check the config file for valid simulation types.')
        

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
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        dt = self.config['time step']
        if dt > 1.:
            self.log.error("Chosen time step too large!")
            exit("Please run with time steps smaller than 1s!")
        self.log.debug('Realistic run')
        self.mc_run = CON_mc_sim(
            self.infected,
            self.pop,
            self.infection,
            self.tracked,
            self.log,
            self.config
        )
        self.t = self.mc_run.time_array
        self.R = self.mc_run.R
        self.log.info('The reproductive number R0 for the run was %.2f' %self.R)
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Finished calculation')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        # Closing log
        self.log.removeHandler(self.fh)
        self.log.removeHandler(self.ch)
        del self.log, self.fh, self.ch
        logging.shutdown()
        return self.mc_run.population


    def __sim_random_walk(self,
              velocity,
              distances,
              seconds=100,
              vel_var=1.,
              dist_var=1.):
        """
        function: __sim_random_walk
        Random walk simulation
        Parameters:
            -float velocity:
                The mean velocity of the current in m/s,
                or the mean "social" velocity 
            -float distances:
                The distances to use. For a social run,
                this is the mean infection distance
            -int seconds:
                Number of seconds to simulate. This is used by
                the mc routines.
            -float vel_var:
                The social velocity variance
            -float dist_var:
                The social distance variance
        Returns:
            -np.array result:
                The resulting infection spread
        """
        dt = self.config['time step']
        if dt > 1.:
            self.log.error("Chosen time step too large!")
            exit("Please run with time steps smaller than 1s!")
        self.log.debug('Random walk run')
        self.t = np.arange(0., seconds, dt)
        self.mc_run = CON_random_walk(
            velocity,
            vel_var,
            distances,
            dist_var,
            self.config['population size'],
            self.infected,
            self.world,
            self.log,
            self.config,
            dt=dt,
            t=self.t
        )
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Finished calculation')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        # Closing log
        self.log.removeHandler(self.fh)
        self.log.removeHandler(self.ch)
        del self.log, self.fh, self.ch
        logging.shutdown()
        return self.mc_run.infections