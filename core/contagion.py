"""
Name: contagion.py
Authors: Stephan Meighen-Berger
Main interface to the contagion module.
This package calculates the spread of an infection in
a given population. It allows for the introduction
of safety measures, such as social distancing and tracking.
"""

# Native modules
import logging
import numpy as np
from time import time
# -----------------------------------------
# Package modules
from .con_config import config
# -----------------------------------------
# Realistic sim imports
from .con_population import CON_population
# -----------------------------------------
# The random walk imports
from .con_adamah import con_adamah
from .con_random_walk import CON_random_walk
from .con_infection import CON_infection
from .con_mc_sim import CON_mc_sim
from .con_measures import CON_measures

logging.basicConfig()

logger = logging.getLogger(__name__)


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
    def __init__(self, infected, rstate=None):
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

        self.rstate = rstate

        pop = config['population size']

        logger.setLevel(logging.DEBUG)

        logger.propagate = False
        # creating file handler with debug messages
        self.fh = logging.FileHandler('../contagion.log', mode='w')
        self.fh.setLevel(logging.DEBUG)

        # Adding the handlers
        logger.addHandler(self.fh)
        logger.info('---------------------------------------------------')
        logger.info('---------------------------------------------------')
        logger.info('Welcome to contagion!')
        logger.info('This package will help you model the spread of infections')
        # Checking the type of the simulation
        logger.info('Simulation type is set to ' + config['simulation type'])
        if config['simulation type'] == 'realistic':
            logger.info('---------------------------------------------------')
            logger.info('---------------------------------------------------')
            logger.info('Starting population construction')
            self.pop = CON_population(pop, rstate=rstate).population
            logger.info('Finished the population')
            logger.info('---------------------------------------------------')
            logger.info('---------------------------------------------------')
            logger.info('Starting the infection construction')
            self.infection = CON_infection(rstate=rstate)
            logger.info('Finished the infection construction')
            logger.info('---------------------------------------------------')
            logger.info('---------------------------------------------------')
            logger.info('Starting the measure construction')
            self.tracked = CON_measures(logger, config).tracked
            logger.info('Finished the measure construction')
            logger.info('---------------------------------------------------')
            logger.info('---------------------------------------------------')
            logger.info('---------------------------------------------------')
            logger.info('---------------------------------------------------')
            logger.info('Setting the simulation framework')
            self.sim = self.__sim_realistic
            logger.info('Simulation framework set. Please type:')
            logger.info('self.sim(parameters) to run the simulation')
            logger.info('---------------------------------------------------')
            logger.info('---------------------------------------------------')
        elif config['simulation type'] == 'random walk':
            logger.info('---------------------------------------------------')
            logger.info('---------------------------------------------------')
            logger.info('Creating the world')
            self.world = con_adamah(logger, config)
            logger.info('Finished world building')
            logger.info('---------------------------------------------------')
            logger.info('---------------------------------------------------')
            logger.info('Setting the simulation framework')
            self.sim = self.__sim_random_walk
            logger.info('Simulation framework set. Please type:')
            logger.info('self.sim(parameters) to run the simulation')
            logger.info('---------------------------------------------------')
            logger.info('---------------------------------------------------')
        else:
            logger.error('Simulation type unknown: ' + config['simulation type'])
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
        logger.info('---------------------------------------------------')
        logger.info('---------------------------------------------------')
        dt = config['time step']
        if dt > 1.:
            logger.error("Chosen time step too large!")
            exit("Please run with time steps smaller than 1s!")
        logger.debug('Realistic run')
        self.mc_run = CON_mc_sim(
            self.infected,
            self.pop,
            self.infection,
            self.tracked,
            rstate=self.rstate
        )
        self.t = self.mc_run.time_array
        self.R = self.mc_run.R

        logger.info('The reproductive number R0 for the run was %.2f' %self.R)
        logger.info('---------------------------------------------------')
        logger.info('---------------------------------------------------')
        logger.info('Finished calculation')
        logger.info('---------------------------------------------------')
        logger.info('---------------------------------------------------')

        return self.mc_run

    def __sim_random_walk(
            self,
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
        dt = config['time step']
        if dt > 1.:
            logger.error("Chosen time step too large!")
            exit("Please run with time steps smaller than 1s!")
        logger.debug('Random walk run')
        self.t = np.arange(0., seconds, dt)
        self.mc_run = CON_random_walk(
            velocity,
            vel_var,
            distances,
            dist_var,
            config['population size'],
            self.infected,
            self.world,
            logger,
            config,
            dt=dt,
            t=self.t
        )
        logger.info('---------------------------------------------------')
        logger.info('---------------------------------------------------')
        logger.info('Finished calculation')
        logger.info('---------------------------------------------------')
        logger.info('---------------------------------------------------')
        # Closing log
        logger.removeHandler(self.fh)
        logger.removeHandler(self.ch)
        del logger, self.fh, self.ch
        logging.shutdown()
        return self.mc_run.infections