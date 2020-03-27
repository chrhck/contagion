# -*- coding: utf-8 -*-

"""
Name: infection.py
Authors: Christian Haack, Stephan Meighen-Berger
Constructs the infection.
"""

# Imports
import numpy as np  # type: ignore
from .pdfs import TruncatedNormal, NormalizedProbability, Beta
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
            self.__log_and_error(
                "Unknown infection probability pdf. Configured: {}".format(
                    config['infection probability pdf']))

        self.__log.debug('The infection duration and incubation pdf')
        if config['infection duration pdf'] == 'gauss':

            dur_pdf = TruncatedNormal(
                0,
                np.inf,
                config['infection duration mean'],
                config['infection duration variance']
                )
            self.__pdf = dur_pdf.rvs

        else:
            self.__log_and_error(
                "Unknown infection duration pdf. Configured: {}".format(
                    config['infection duration pdf']))

        if config['infectious duration pdf'] == 'gauss':

            dur_pdf = TruncatedNormal(
                0,
                np.inf,
                config['infectious duration mean'],
                config['infectious duration variance']
                )
            self.__infectious_duration = dur_pdf.rvs

        else:
            self.__log_and_error(
                "Unknown infectious duration pdf. Configured: {}".format(
                    config['infectious duration pdf']))

        if config['incubation duration pdf'] == 'gauss':
            dur_pdf = TruncatedNormal(
                0,
                np.inf,
                config['incubation duration mean'],
                config['incubation duration variance']
                )
            self.__incubation_duration = dur_pdf.rvs
        else:
            self.__log_and_error(
                "Unknown incubation duration pdf. Configured: {}".format(
                    config['incubation duration pdf']))

        if config['recovery time pdf'] == 'gauss':
            recovery_time_pdf = TruncatedNormal(
                0,
                np.inf,
                config['recovery time mean'],
                config['recovery time sd']
                )
            self.__recovery_time = recovery_time_pdf.rvs
        else:
            self.__log_and_error(
                "Unknown recovery time pdf. Configured: {}".format(
                    config['recovery time pdf']))


        if config['hospitalization probability pdf'] == 'beta':
            hospit_prob_pdf = Beta(
                config['hospitalization probability mean'],
                config['hospitalization probability sd']
                )
            self._hospitalization_prob = hospit_prob_pdf.rvs
        else:
            self.__log_and_error(
                "Unknown hospitalization probability pdf. Configured: {}".format(
                    config['hospitalization probability pdf']))

        if config['hospitalization duration pdf'] == 'gauss':
            hospit_dur_pdf = TruncatedNormal(
                0,
                np.inf,
                config['hospitalization duration mean'],
                config['hospitalization duration sd']
                )
            self.__hospitalization_duration = hospit_dur_pdf.rvs
        else:
            self.__log_and_error(
                "Unknown hospitalization duration pdf. Configured: {}".format(
                    config['hospitalization duration pdf']))

        if config['time until hospitalization pdf'] == 'gauss':
            hospit_dur_until_pdf = TruncatedNormal(
                0,
                np.inf,
                config['time until hospitalization mean'],
                config['time until hospitalization sd']
                )
            self.__time_until_hospitalization = hospit_dur_until_pdf.rvs
        else:
            self.__log_and_error(
                "Unknown hospitalization duration pdf. Configured: {}".format(
                    config['hospitalization duration pdf']))

        if config['time incubation death pdf'] == 'gauss':
            time_till_death_pdf = TruncatedNormal(
                0,
                np.inf,
                config['time incubation death mean'],
                config['time incubation death sd']
                )
            self.__time_incubation_death = time_till_death_pdf.rvs
        else:
            self.__log_and_error(
                "Unknown hospitalization duration pdf. Configured: {}".format(
                    config['hospitalization duration pdf']))

        if config['mortality prob pdf'] == 'beta':
            death_prob_pdf = Beta(
                config['mortality rate mean'],
                config['mortality rate sd']
                )
            self.__death_prob = death_prob_pdf.rvs
        else:
            self.__log_and_error(
                "Unknown hospitalization duration pdf. Configured: {}".format(
                    config['hospitalization duration pdf']))

    # TODO: Remove this
    def __log_and_error(self, msg):
        """
        function: __log_and_error
        Custom log error message for this file
        Parameters:
            -str msg:
                The error message
        Returns:
            -RuntimeError with the message
        """
        self.__log.error(msg)
        raise RuntimeError(msg)

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
    def incubation_duration(self):
        """
        function: incubation_duration
        Getter function for the incubation duration
        duration
        Parameters:
            -None
        Returns:
            -int incubation_duration:
                The duration of incubation
        """
        return self.__incubation_duration

    @property
    def infectious_duration(self):
        """
        function: infectious_duration
        Getter function for the incubation duration
        duration
        Parameters:
            -None
        Returns:
            -int infectious_duration:
                The duration of infection
        """
        return self.__infectious_duration

    @property
    def hospitalization_prob(self):
        """
        function: hospitalization_prob
        Getter function for the hospitalization probability
        duration
        Parameters:
            -None
        Returns:
            -hospitalization_prob:
                The probability of hospitalization
        """
        return self._hospitalization_prob

    @property
    def time_until_hospitalization(self):
        """
        function: time_until_hospitalization
        Getter function for the time until hospit.
        duration
        Parameters:
            -None
        Returns:
            -time_until_hospitalization:
                Time until hospitalization
        """
        return self.__time_until_hospitalization

    @property
    def hospitalization_duration(self):
        """
        function: hospitalization_duration
        Getter function for the hospit. length
        duration
        Parameters:
            -None
        Returns:
            -hospitalization_duration:
                The duration of hospit.
        """
        return self.__hospitalization_duration

    @property
    def recovery_time(self):
        """
        function: recovery_time
        Getter function for the recovery duration
        duration
        Parameters:
            -None
        Returns:
            -recovery_time:
                The duration of recovery
        """
        return self.__recovery_time

    @property
    def death_prob(self):
        """
        function: death_prob
        Getter function for the death probability
        duration
        Parameters:
            -None
        Returns:
            -death_prob:
                The probability of death
        """
        return self.__death_prob

    @property
    def time_incubation_death(self):
        """
        function: time_incubation_death
        Getter function for the time of incubation and death
        duration
        Parameters:
            -None
        Returns:
            -time_incubation_death:
                Incubation and death time
        """
        return self.__time_incubation_death
