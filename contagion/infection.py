# -*- coding: utf-8 -*-

"""
Name: infection.py
Authors: Christian Haack, Stephan Meighen-Berger
Constructs the infection.
"""

# Imports
import numpy as np  # type: ignore
from .pdfs import TruncatedNormal, NormalizedProbability
from .config import config


class Infection(object):
    """
    class: Infection
    Constructs the infection object
    Parameters:
        -obj log:
            The logger
    Returns:
        -None
    """
    def __init__(self, log):
        """
        function: __init__
        Initializes the infection object
        Parameters:
            -obj log:
                The logger
        Returns:
            -None
        """
        # TODO: Set up standard parameters for different diseases, which
        #   can be loaded by only setting the disease
        self.__log = log.getChild(self.__class__.__name__)

        if config['random state'] is None:
            self.__log.warning("No random state given, constructing new state")
            self.__rstate = np.random.RandomState()
        else:
            self.__rstate = config['random state']

        self.__log.debug('The infection probability pdf')
        if config['infection probability pdf'] == 'intensity':
            self.__pdf_infection_prob = NormalizedProbability(0, 1)
        else:
            self.__log.error('Unrecognized infection pdf! Set to ' +
                             config['infection probability pdf'])
            exit('Check the infection probability pdf in the config file!')

        self.__log.debug('The infection duration and incubation pdf')
        if config['infection duration pdf'] == 'gauss':

            dur_pdf = TruncatedNormal(
                0,
                np.inf,
                config['infection duration mean'],
                config['infection duration variance']
                )
            self._pdf = dur_pdf.rvs

        else:
            self.__log.error('Unrecognized infection duration pdf! Set to ' +
                             config['infection duration pdf'])
            exit('Check the infection duration pdf in the config file!')

        if config['infectious duration pdf'] == 'gauss':

            dur_pdf = TruncatedNormal(
                0,
                np.inf,
                config['infectious duration mean'],
                config['infectious duration variance']
                )
            self._infectious_duration = dur_pdf.rvs

        else:
            self.__log.error('Unrecognized infectious duration pdf! Set to ' +
                             config['infection duration pdf'])
            exit('Check the infectious duration pdf in the config file!')

        if config['incubation duration pdf'] == 'gauss':

            dur_pdf = TruncatedNormal(
                0,
                np.inf,
                config['incubation duration mean'],
                config['incubation duration variance']
                )
            self._incubation_duration = dur_pdf.rvs

        else:
            self.__log.error('Unrecognized incubation duration pdf! Set to ' +
                             config['infection duration pdf'])
            exit('Check the incubation duration pdf in the config file!')

    @property
    def pdf_infection_prob(self):
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
        return self.__pdf_infection_prob

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

    @property
    def immunity_dur(self):
        """
        function: immunity_dur
        Getter function for the immunity
        duration
        Parameters:
            -None
        Returns:
            -int length:
                The duration of immunity
        """

    @property
    def incubation_duration(self):
        return self._incubation_duration

    @property
    def infectious_duration(self):
        return self._infectious_duration
