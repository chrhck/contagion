"""
Name: con_infection.py
Authors: Stephan Meighen-Berger
Constructs the infection.
"""

# Imports
import numpy as np
from scipy.stats import truncnorm

class CON_infection(object):
    """
    class: CON_infection
    Constructs the infection object
    Parameters:
        -obj log:
            The logger
        -dic config:
            The configuration dictionary
    Returns:
        -None
    """
    def __init__(self, log, config, rstate=None):
        """
        function: __init__
        Initializes the infection object
        Parameters:
            -obj log:
                The logger
            -dic config:
                The configuration dictionary
        Returns:
            -None
        """
        #TODO: Set up standard parameters for different diseases, which
        #   can be loaded by only setting the disease
        self.__log = log
        self.__config = config

        if rstate is None:
            self.__log.warning("No random state given, constructing new state")
            rstate = np.random.RandomState()
        self.__rstate = rstate

        self.__log.debug('The infection probability pdf')
        if self.__config['infection probability pdf'] == 'intensity':
            self.__pdf = self.__pdf_intensity
        else:
            self.__log.error('Unrecognized infection pdf! Set to ' +
                             self.__config['infection probability pdf'])
            exit('Check the infection probability pdf in the config file!')

        self.__log.debug('The infection duration pdf')
        if self.__config['infection duration pdf'] == 'gauss':
            self.__pdf_duration = self.__pdf_duration_norm
        else:
            self.__log.error('Unrecognized infection duration pdf! Set to ' +
                             self.__config['infection duration pdf'])
            exit('Check the infection duration pdf in the config file!')

    @property
    def pdf(self):
        """
        function: pdf
        The infection probability
        Parameters:
            -None
        Returns:
            -function __pdf
                The infection probability
                Takes the sc intensity
        """
        return self.__pdf

    @property
    def pdf_duration(self):
        """
        function: pdf_duration
        The duration of the infection
        Parameters:
            -None
        Returns:
            -function pdf_duration
                Takes an int
        """
        return self.__pdf_duration

    def __pdf_intensity(self, intensity):
        """
        function: __pdf_intensity
        The infection probability is equal to the intensity of the interaction
        Parameters:
            -float intensity:
                The social interaction intensity
        Returns:
            -float prob:
                The infection probability
        """
        prob = intensity
        return prob

    def __pdf_duration_norm(self, infected):
        """
        function: __pdf_duration_norm
        The normal distribution for the length of infection
        Parameters:
            -int infected:
                The number of infected people
        Returns:
            -np.array duration:
                The length of their infection
        """

        mean = self.__config['infection duration mean']
        scale = self.__config['infection duration variance']

        # Minimum social circle size is 0
        a, b = (0 - mean) / scale, (np.inf - mean) / scale

        # could also use binomial here
        duration = truncnorm.rvs(
            a, b, loc=mean, scale=scale, size=infected,
            random_state=self.__rstate)
        # duration = np.asarray(duration, dtype=int)

        return duration
