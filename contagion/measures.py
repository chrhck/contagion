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

        self.__contact_tracing = config["measures"]["contact tracing"]
        self.__quarantine = config["measures"]["quarantine"]
        self.__testing = config["measures"]["testing"]

        if self.__contact_tracing:
            _log.info("Using contact tracing")
            self.__def_contact_tracing()
        else:
            _log.info("No contact tracing")
            self.__tracked = None
            # self.__backtrack_length = None
            # self.__second_order_tracing = None
            # self.__track_uninfected = None

        if self.__quarantine:
            _log.info("Using quarantine")
            self.__def_quarantine_pdf()
        else:
            _log.info("No quarantine")
            # self.__quarantine_duration = None

        if self.__testing:
            _log.info("Using testing")
            self.__def_testing()
        else:
            _log.info("No testing")
            # self.__time_until_test_pdf = None
            # self.__time_until_test_result_pdf = None
            # self.__test_efficiency_function = None

    @property
    def contact_tracing(self):
        return self.__contact_tracing

    @property
    def quarantine(self):
        return self.__quarantine

    @property
    def testing(self):
        return self.__testing

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
        return self.__test_efficiency_function(points)

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
    def track_uninfected(self):
        """
        function: backtrack_length
        Getter function for the length of the contact backtracing
        Parameters:
            -None
        Returns:
            -Int
        """
        return np.bool(self.__track_uninfected)

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
    def __def_contact_tracing(self):
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
        _log.debug(
            "Length of backtracking: {0}".format(self.__backtrack_length)
        )

        self.__second_order_tracing = config["measures"]["second order"]
        _log.debug(
            "Second order tracing: {0}".format(self.__second_order_tracing)
        )

        self.__track_uninfected = config["measures"]["track uninfected"]
        _log.debug("Track uninfected: {0}".format(self.__second_order_tracing))

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

        self.__test_efficiency_function = np.vectorize(
            test_efficiency_function
        )


# Function fitted for the test efficiency:
# x : days after infectious
# a = 8.5
# b = 0.74408163
# c = 8.48725228
def test_efficiency_function(x, a=8.5, b=0.74408163, c=8.48725228):
    if x < a:
        return 0.95
    else:
        return np.exp(-b * (x - c))
