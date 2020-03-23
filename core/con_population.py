"""
Name: con_population.py
Authors: Stephan Meighen-Berger
Constructs the population.
"""

"Imports"
from sys import exit
import numpy as np
from scipy.stats import norm

class CON_population(object):
    """
    class: CON_pupulation
    Class to help with the construction of a realistic population
    Paremeters:
        -int pop:
            The size of the populations
        -obj log:
            The logger
        -dic config:
            The dictionary from the config file
    Returns:
        -None
    """
    def __init__(self, pop, log, config):
        """
        function: __init__
        Initializes the population
        Parameters:
            -int pop:
                The size of the population
            -obj log:
                The logger
            -dic config:
                The dictionary from the config file
        Returns:
            -None
        """
        self.__config = config
        self.__log = log
        self.__log.info('Constructing social circles for the population')

        self.__log.debug('Number of people in social circles')
        if self.__config['social circle pdf'] == 'gauss':
            self.__social_circles = self.__social_pdf_norm(pop)
        else:
            self.__log.error('Unrecognized social pdf! Set to ' + self.__config['social circle pdf'])
            exit('Check the social circle distribution in the config file!')

        self.__log.debug('The social circle interactions for each person')
        if self.__config['social circle interactions pdf'] == 'gauss':
            self.__sc_interactions = self.__sc_interact_norm(pop)
        else:
            self.__log.error('Unrecognized sc interactions pdf! Set to ' +
                             self.__config['social circle interactions pdf'])
            exit('Check the social circle interactions distribution in the config file!')

        self.__log.debug('Constructing population')
        self.__pop = np.array([
            [np.random.randint(0, high=pop, size=self.__social_circles[i]),
             self.__sc_interactions[i]]
            for i in range(pop)
        ])

    @property
    def population(self):
        """
        function: population
        Returns the population
        Parameters:
            -None
        Returns:
            -np.array population:
                The constructed population
        """
        return self.__pop

    def __social_pdf_norm(self, pop):
        """
        function: __social_pdf_norm
        Constructs the number of people in each person's circle
        Parameters:
            -int pop:
                The population size
            -np.array circles:
                The number of people in each circle
        Returns:
            -np.array circles:
                The size of the social circles for every individual
        """
        # Re-rolling if negative
        circles = []
        for _ in range(pop):
            res = -1
            while res < 0:
                res = int(norm.rvs(
                    size=1,
                    loc=self.__config['average social circle'],
                    scale=self.__config['variance social circle']))
            circles.append(res)
        return(np.array(circles))

    def __sc_interact_norm(self, pop):
        """
        function: __sc_interact_norm
        The number of interactions within each person's circle
        Parameters:
            -int pop:
                The size of the population
        Returns:
            -np.array interact:
                The number of interactions within the social circle
        """
        interact = []
        # Re-rolling negative values
        for i in range(pop):
            res = -1
            while res < 0:
                res = int(norm.rvs(
                    size=1,
                    loc=self.__config['mean social circle interactions'],
                    scale=self.__config['variance social circle interactions']
                ))
                # Never interact more than the size of the sc
                if res > self.__social_circles[i]:
                    res = -1
            interact.append(res)
        return(np.array(interact))

