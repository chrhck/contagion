# -*- coding: utf-8 -*-

"""
Name: measures.py
Authors: Stephan Meighen-Berger
The different measures one can take
to suppress the spread.
"""

# imports
from sys import exit
import numpy as np
import logging

from .pdfs import Uniform
from .config import config

_log = logging.getLogger(__name__)


class Measures(object):
    """
    class: Measures
    Class to implement different possible
    containment measures.
    Parameters:
        -None
    Returns:
        -None
    """

    def __init__(self):
        """
        function: __init__
        Initializes the class
        Parameters:
            -None
        Returns:
            -None
        """
        if config["runtime"]["random state"] is None:
            _log.warning("No random state given, constructing new state")
            self.__rstate = np.random.RandomState()
        else:
            self.__rstate = config["runtime"]["random state"]

        if config["measures"]["type"] == None:
            _log.info("No measure taken")
            self.__tracked = None
            self.__distanced = None
        elif config["measures"]["type"] == "contact_tracing":
            _log.info("Using contact tracing")
            self.__contact_tracing()
            self.__distanced = None
        elif config["measures"]["type"] == "social_distancing":
            _log.info("Using social distancing")
            self.__social_distancing()
            self.__tracked = None
        elif config["measures"]["type"] == "all":
            _log.info("Using social distancing")
            _log.info("Using contact tracing")
            self.__contact_tracing()
            self.__social_distancing()
        else:
            _log.error("measure not implemented!")
            exit("Please check the config file what measures are allowed")

        self.__def_quarantine_pdf()

    @property
    def tracked(self):
        """
        function: tracked
        Getter function for the tracked population
        Parameters:
            -None
        Returns:
            -np.array tracked:
                The ids of the tracked population
        """
        return self.__tracked

    @property
    def distanced(self):
        """
        function: distanced
        Getter function for the distanced population
        Parameters:
            -None
        Returns:
            -np.array distanced:
                The ids of the distanced population
        """
        return self.__distanced

    @property
    def quarantine_duration(self):
        """
        function: quarantine_duration
        Getter function for the duration of the quarantine
        Parameters:
            -None
        Returns:
            -PDF
                The pdf of the quarantine duration
        """
        return self.__quarantine_duration

    # TODO: Not 100% of participants will report correctly
    def __contact_tracing(self):
        """
        function: __contact_tracing
        Implements the measure contact tracing
        Parameters:
            -None
        Returns:
            -None
        """
        tracked_pop = int(
            config["population"]["population size"]
            * config["measures"]["tracked fraction"]
        )
        _log.debug("Number of people tracked is %d" % tracked_pop)
        self.__tracked = np.random.choice(
            range(config["population"]["population size"]),
            size=tracked_pop,
            replace=False,
        ).flatten()

    def __social_distancing(self):
        """
        function: __social_distancing
        Implements the measure social distancing
        Parameters:
            -None
        Returns:
            -None
        """
        distanced_pop = int(
            config["population"]["population size"]
            * config["measures"]["distanced fraction"]
        )
        _log.debug("Number of people social distancing is %d" % distanced_pop)
        self.__distanced = np.random.choice(
            range(config["population"]["population size"]),
            size=distanced_pop,
            replace=False,
        ).flatten()

    def __def_quarantine_pdf(self):
        """
        function: __def_quarantine_pdf
        Defines the pdf for the duration of the quarantine
        Parameters:
            -None
        Returns:
            -None
        """
        quarantine_duration_pdf = Uniform(
            config["measures"]["quarantine duration"], 0.0
        )
        self.__quarantine_duration = quarantine_duration_pdf
