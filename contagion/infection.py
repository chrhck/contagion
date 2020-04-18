# -*- coding: utf-8 -*-

"""
Name: infection.py
Authors: Christian Haack, Stephan Meighen-Berger
Constructs the infection.
"""

# Imports
import logging

from .config import config
from .pdfs import construct_pdf

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
        self._rstate = config["runtime"]["random state"]

        # Infection probability
        _log.debug("The infection probability pdf")

        infection_prob_pdf = construct_pdf(
            config["infection"]["infection probability pdf"])

        self.__pdf_infection_prob = infection_prob_pdf

        _log.debug("The infection duration pdf")

        # Infectious duration
        dur_infectious_pdf = construct_pdf(
            config["infection"]["infectious duration pdf"])
        self.__infectious_duration = dur_infectious_pdf

        # Incubation duration
        _log.debug("The incubation duration pdf")
        dur_incubation_pdf = construct_pdf(
            config["infection"]["incubation duration pdf"])
        self.__incubation_duration = dur_incubation_pdf

        # Latency
        _log.debug("The latency duration pdf")
        dur_latent_pdf = construct_pdf(
            config["infection"]["latency duration pdf"])
        self.__latent_duration = dur_latent_pdf

        # Recovery
        _log.debug("The recovery time pdf")

        recovery_time_pdf = construct_pdf(
            config["infection"]["recovery time pdf"])
        self.__recovery_time = recovery_time_pdf

        # Symptoms
        _log.debug("Symptoms pdf")

        will_have_symptoms_pdf = construct_pdf(
            config["infection"]["will have symptoms prob pdf"])

        self.__will_have_symptoms_prob = will_have_symptoms_pdf

        # Hospitalization
        _log.debug("The hospitalization pdfs")

        hospit_prob_pdf = construct_pdf(
            config["infection"]["hospitalization probability pdf"])
        self._hospitalization_prob = hospit_prob_pdf

        hospit_dur_pdf = construct_pdf(
            config["infection"]["hospitalization duration pdf"])
        self.__hospitalization_duration = hospit_dur_pdf

        hospit_dur_until_pdf = construct_pdf(
            config["infection"]["time until hospitalization pdf"])
        self.__time_until_hospitalization = hospit_dur_until_pdf

        # Mortality
        _log.debug("The mortality pdfs")

        time_till_death_pdf = construct_pdf(
            config["infection"]["time incubation death pdf"])
        self.__time_incubation_death = time_till_death_pdf

        death_prob_pdf = construct_pdf(
            config["infection"]["mortality prob pdf"])
        self.__death_prob = death_prob_pdf

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
    def latent_duration(self):
        """
        function: latent_duration
        Getter function for the latent duration
        duration
        Parameters:
            -None
        Returns:
            -int latent_duration:
                The duration of latent period
        """
        return self.__latent_duration

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

    @property
    def will_have_symptoms_prob(self):

        return self.__will_have_symptoms_prob
