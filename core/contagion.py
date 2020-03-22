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
# Package modules
from con_config import config
from con_adamah import con_adamah
from con_mc_sim import con_mc_sim

class CONTAGION(object):
    """
    class: CONTAGION
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
        Initializes the class CONTAGION.
        Here all run parameters are set.
        Parameters:
            -None
        Returns:
            -None
        """
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
        self.ch.setLevel(config['debug level'])
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
        # TODO: Population model
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Creating the world')
        # TODO: Social circle models for the "world"
        # TODO: Encounter distribution models
        #  The volume of interest
        self.world = con_adamah(self.log)
        self.log.info('Finished world building')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')

    def solve(self, population, velocity,
              distances,
              seconds=100,
              dt=config['time step'],
              vel_var=1.,
              dist_var=1.,
              infected=1):
        """
        function: solve
        Calculates the light yields depending on input
        Parameters:
            -float population:
                The number of organisms
            -float velocity:
                The mean velocity of the current in m/s,
                or the mean "social" velocity 
            -float distances:
                The distances to use. For a social run,
                this is the mean infection distance
            -int seconds:
                Number of seconds to simulate. This is used by
                the mc routines.
            -float regen:
                The regeneration factor
            -float dt:
                The time step to use. Needs to be below 1
            -float vel_var:
                The social velocity variance
            -float dist_var:
                The social distance variance
            -int infected:
                The number of infected people
        Returns:
            -np.array result:
                The resulting light yields
        """
        if dt > 1.:
            self.log.error("Chosen time step too large!")
            exit("Please run with time steps smaller than 1s!")
        self.log.debug('Monte-Carlo run')
        self.t = np.arange(0., seconds, dt)
        self.mc_run = con_mc_sim(
            velocity,
            vel_var,
            distances,
            dist_var,
            population,
            infected,
            self.world,
            self.log,
            dt=dt,
            t=self.t
        )
        self.log.debug('---------------------------------------------------')
        self.log.debug('---------------------------------------------------')
        self.log.info('Finished calculation')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        return self.mc_run.infections