# -*- coding: utf-8 -*-

"""
Name: infection.py
Authors: Christian Haack, Stephan Meighen-Berger
Constructs the infection.
"""

# Imports
import numpy as np  # type: ignore
from .pdfs import TruncatedNormal, NormalizedProbability, Beta, Gamma
import logging
from .config import config

_log = logging.getLogger(__name__)


class Infection(object):
    """
    class: Infection
    Constructs the infection object
    Parameters:
        -None
    Returns:
        -None
    """
    def __init__(self):
        """
        function: __init__
        Initializes the infection object
        Parameters:
            -None
        Returns:
            -None
        """
        # TODO: Set up standard parameters for different diseases, which
        #   can be loaded by only setting the disease
        if config['random state'] is None:
            _log.warning("No random state given, constructing new state")
            self.__rstate = np.random.RandomState()
        else:
            self.__rstate = config['random state']

        _log.debug('The infection probability pdf')
        if config['infection probability pdf'] == 'intensity':
            self.__pdf_infection_prob = NormalizedProbability(0, 1)
        else:
            _log.error('Unrecognized infection pdf! Set to ' +
                             config['infection probability pdf'])
            exit('Check the infection probability pdf in the config file!')

        _log.debug('The infection pdfs')
        if config['infection duration pdf'] == 'gauss':
            dur_pdf = TruncatedNormal(
                0,
                np.inf,
                config['infection duration mean'],
                config['infection duration variance']
                )
            self.__pdf = dur_pdf
        elif config['infection duration pdf'] == 'gamma':
            dur_pdf = Gamma(
                config['infection duration mean'],
                config['infection duration variance']
            )
            self.__pdf = dur_pdf
        else:
            _log.error('Unrecognized infection duration pdf! Set to ' +
                             config['infection duration pdf'])
            exit('Check the infection duration pdf in the config file!')

        if config['infectious duration pdf'] == 'gauss':
            dur_pdf = TruncatedNormal(
                0,
                np.inf,
                config['infectious duration mean'],
                config['infectious duration variance']
                )
            self.__infectious_duration = dur_pdf
        elif config['infectious duration pdf'] == 'gamma':
            dur_pdf = Gamma(
                config['infectious duration mean'],
                config['infectious duration variance']
            )
            self.__infectious_duration = dur_pdf
        else:
            _log.error('Unrecognized infectious duration pdf! Set to ' +
                             config['infection duration pdf'])
            exit('Check the infectious duration pdf in the config file!')

# TODO: Update logger messages
        if config['incubation duration pdf'] == 'gauss':
            dur_pdf = TruncatedNormal(
                0,
                np.inf,
                config['incubation duration mean'],
                config['incubation duration variance']
                )
            self.__incubation_duration = dur_pdf
        elif config['incubation duration pdf'] == 'gamma':
            dur_pdf = Gamma(
                config['incubation duration mean'],
                config['incubation duration variance']
            )
            self.__incubation_duration = dur_pdf
        else:
            _log.error('Unrecognized infectious duration pdf! Set to ' +
                             config['infection duration pdf'])
            exit('Check the infectious duration pdf in the config file!')

# TODO: Update logger messages
        if config['recovery time pdf'] == 'gauss':
            recovery_time_pdf = TruncatedNormal(
                0,
                np.inf,
                config['recovery time mean'],
                config['recovery time sd']
                )
            self.__recovery_time = recovery_time_pdf
        elif config['recovery time pdf'] == 'gamma':
            recovery_time_pdf = Gamma(
                config['recovery time mean'],
                config['recovery time sd']
            )
            self.__recovery_time = recovery_time_pdf
        else:
            _log.error('Unrecognized infectious duration pdf! Set to ' +
                             config['infection duration pdf'])
            exit('Check the infectious duration pdf in the config file!')


# TODO: Update logger messages
        if config['hospitalization probability pdf'] == 'beta':
            hospit_prob_pdf = Beta(
                config['hospitalization probability mean'],
                config['hospitalization probability sd']
                )
            self._hospitalization_prob = hospit_prob_pdf
        else:
            _log.error('Unrecognized infectious duration pdf! Set to ' +
                             config['infection duration pdf'])
            exit('Check the infectious duration pdf in the config file!')

        if config['hospitalization duration pdf'] == 'gauss':
            hospit_dur_pdf = TruncatedNormal(
                0,
                np.inf,
                config['hospitalization duration mean'],
                config['hospitalization duration sd']
                )
            self.__hospitalization_duration = hospit_dur_pdf
        elif config['hospitalization duration pdf'] == 'gamma':
            hospit_dur_pdf = Gamma(
                config['hospitalization duration mean'],
                config['hospitalization duration sd']
            )
            self.__hospitalization_duration = recovery_time_pdf
        else:
            _log.error('Unrecognized infectious duration pdf! Set to ' +
                             config['infection duration pdf'])
            exit('Check the infectious duration pdf in the config file!')

# TODO: Update logger messages
        if config['time until hospitalization pdf'] == 'gauss':
            hospit_dur_until_pdf = TruncatedNormal(
                0,
                np.inf,
                config['time until hospitalization mean'],
                config['time until hospitalization sd']
                )
            self.__time_until_hospitalization = hospit_dur_until_pdf
        elif config['time until hospitalization pdf'] == 'gamma':
            hospit_dur_until_pdf = Gamma(
                config['time until hospitalization mean'],
                config['time until hospitalization sd']
            )
            self.__time_until_hospitalization = hospit_dur_until_pdf
        else:
            _log.error('Unrecognized infectious duration pdf! Set to ' +
                             config['infection duration pdf'])
            exit('Check the infectious duration pdf in the config file!')

# TODO: Update logger messages
        if config['time incubation death pdf'] == 'gauss':
            time_till_death_pdf = TruncatedNormal(
                0,
                np.inf,
                config['time incubation death mean'],
                config['time incubation death sd']
                )
            self.__time_incubation_death = time_till_death_pdf
        elif config['time incubation death pdf'] == 'gamma':
            time_till_death_pdf = Gamma(
                config['time incubation death mean'],
                config['time incubation death sd']
            )
            self.__time_incubation_death = time_till_death_pdf
        else:
            _log.error('Unrecognized infectious duration pdf! Set to ' +
                             config['infection duration pdf'])
            exit('Check the infectious duration pdf in the config file!')

# TODO: Update logger messages
        if config['mortality prob pdf'] == 'beta':
            death_prob_pdf = Beta(
                config['mortality rate mean'],
                config['mortality rate sd']
                )
            self.__death_prob = death_prob_pdf
        else:
            _log.error('Unrecognized incubation duration pdf! Set to ' +
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
