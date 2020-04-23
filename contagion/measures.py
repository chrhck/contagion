# -*- coding: utf-8 -*-

"""
Name: measures.py
Authors: Stephan Meighen-Berger, Andrea Turcati
The different measures one can take
to suppress the spread.
"""

# imports
from sys import exit
from typing import Union, Optional
import numpy as np
from scipy.interpolate import interp1d
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
        self.__rstate = config["runtime"]["random state"]

        if config["measures"]["type"] is None:
            _log.info("No measure taken")
            self.__tracked = None
        elif config["measures"]["type"] == "contact_tracing":
            _log.info("Using contact tracing")
            self.__contact_tracing()
        else:
            _log.error("measure not implemented!")
            exit("Please check the config file what measures are allowed")

        self.__def_quarantine_pdf()

        self.__def_testing()

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

    @property
    def time_until_test(self):
        """
        function: time_until_test
        Getter function for the delay of the testing
        Parameters:
            -None
        Returns:
            -PDF
                The pdf of the quarantine duration
        """
        return self.__time_until_test_pdf

    @property
    def time_until_test_result(self):
        """
        function: time_until_test_result
        Getter function for the delay of the test results
        Parameters:
            -None
        Returns:
            -PDF
                The pdf of the quarantine duration
        """
        return self.__time_until_test_result_pdf

    def test_efficiency(
        self, points: Union[float, np.ndarray], dtype: Optional[type] = None
    ) -> np.ndarray:
        return self.__test_efficiency(points)

    @property
    def backtrack_length(self):
        """
        function: backtrack_length
        Getter function for the length of the contact backtracing
        Parameters:
            -None
        Returns:
            -Int
        """
        return np.int(self.__backtrack_length)

    @property
    def is_SOT_active(self):
        """
        function: is_SOT_active
        Getter function for the Second Order Tracing
        Parameters:
            -None
        Returns:
            -Bool
        """
        return self.__second_order_tracing

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
        self.__tracked = self.__rstate.choice(
            range(config["population"]["population size"]),
            size=tracked_pop,
            replace=False,
        ).flatten()

        self.__backtrack_length = config["measures"]["backtrack length"]

        if type(config["measures"]["second order"]) == np.bool:
            self.__second_order_tracing = config["measures"]["second order"]
            _log.debug(
                "Second order tracing: {0}".format(self.__second_order_tracing)
            )
        else:
            _log.error("Second order tracing must be True or False.")
            exit("Please check the config file.")

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

    def __def_testing(self):
        """
        function: __def_testing
        Defines the testing parameters
        Parameters:
            -None
        Returns:
            -None
        """
        time_until_test_pdf = Uniform(
            config["measures"]["time until test"], 0.0
        )
        self.__time_until_test_pdf = time_until_test_pdf

        time_until_test_result_pdf = Uniform(
            config["measures"]["time until result"], 0.0
        )
        self.__time_until_test_result_pdf = time_until_test_result_pdf

        f_eff = np.load(
            "../data/test_efficiency/test_efficiency_vs_infectious_duration.npy"
        )
        f_int = interp1d(f_eff["x"], f_eff["y"], kind="linear")
        self.__test_efficiency = f_int
