# -*- coding: utf-8 -*-

"""
Name: measures.py
Authors: Stephan Meighen-Berger, Andrea Turcati
The different measures one can take
to suppress the spread.
"""

# imports
from typing import Union, Optional
import numpy as np
import logging

from .pdfs import Uniform, Delta, construct_pdf
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
        self.__population_tracking = config["measures"]["population tracking"]
        self.__quarantine = config["measures"]["quarantine"]
        self.__testing = config["measures"]["testing"]
        self.__rnd_testing = config["measures"]["rnd testing"]

        if self.__contact_tracing:
            _log.info("Using contact tracing")
            self.__def_contact_tracing()
        else:
            _log.info("No contact tracing")

        if self.__population_tracking:
            _log.info("Using population tracking")
        else:
            _log.info("No population tracking")

        if self.__quarantine:
            _log.info("Using quarantine")
            self.__def_quarantine()
        else:
            _log.info("No quarantine")

        if self.__testing:
            _log.info("Using testing")
            self.__def_testing()
        else:
            _log.info("No testing")

        if self.__rnd_testing:
            self.__def_rnd_testing()

        self.measures_active = True

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
    def population_tracking(self):
        return self.__population_tracking

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
    def report_symptomatic(self):
        return self.__report_symptomatic

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
    def time_until_second_test(self):
        """
        Getter function for the delay of the second test
        Parameters:
            -None
        Returns:
            -PDF
        """
        return self.__time_until_second_test_pdf

    @property
    def time_until_second_test_result(self):
        """
        Getter function for the delay of the second test
        Parameters:
            -None
        Returns:
            -PDF
        """
        return self.__time_until_second_test_result_pdf

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

    @property
    def test_false_positive_pdf(self):
        """
        function: time_until_test_result
        Getter function for the delay of the test results
        Parameters:
            -None
        Returns:
            -PDF
                The pdf of the quarantine duration
        """
        return self.__test_false_positive_pdf

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
    def tracing_efficiency(self):
        return self.__tracing_efficiency

    @property
    def random_test_num(self):
        return self.__random_test_num

    @property
    def random_testing(self):
        return self.__rnd_testing

    @property
    def random_test_mode(self):
        return self.__random_test_mode

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

        self.__tracing_efficiency = config["measures"]["tracing efficiency"]


    def __def_quarantine(self):
        """
        function: __def_quarantine_pdf
        Defines the pdf for the duration of the quarantine
        Parameters:
            -None
        Returns:
            -None
        """
        quarantine_duration_pdf = Delta(
            config["measures"]["quarantine duration"])
        self.__quarantine_duration = quarantine_duration_pdf
        self.__report_symptomatic = config["measures"]["report symptomatic"]

    def __def_rnd_testing(self):
        self.__random_test_num = config["measures"]["random test num"]
        self.__random_test_mode = config["measures"]["testing mode"]

    def __def_testing(self):
        """
        function: __def_testing
        Defines the testing parameters
        Parameters:
            -None
        Returns:
            -None
        """

        # Function fitted for the test efficiency:
        # x : days after infectious
        # a = 8.5
        # b = 0.74408163
        # c = 8.48725228

        infectivity_curve = construct_pdf(
            config["infection"]["infection probability pdf"])

        def test_efficiency_function(x):
            x = np.atleast_1d(x)
            infectivity = infectivity_curve.pdf(x)

            tpr = config["measures"]["test true positive rate"]
            t_thresh = config["measures"]["test threshold"]

            efficiency = np.empty_like(infectivity)

            mask = infectivity > t_thresh
            efficiency[mask] = tpr
            efficiency[~mask] = tpr/t_thresh * infectivity[~mask]

            return efficiency

        time_until_test_pdf = Delta(config["measures"]["time until test"])
        self.__time_until_test_pdf = time_until_test_pdf

        time_until_test_result_pdf = Delta(
            config["measures"]["time until result"]
        )

        self.__time_until_test_result_pdf = time_until_test_result_pdf

        time_until_second_test_pdf = Delta(
            config["measures"]["time until second test"]
        )
        self.__time_until_second_test_pdf = time_until_second_test_pdf

        time_until_second_test_result_pdf = Delta(
            config["measures"]["time until second test result"]
        )
        self.__time_until_second_test_result_pdf =\
            time_until_second_test_result_pdf

        self.__test_efficiency_function = test_efficiency_function

        self.__test_false_positive_pdf = Delta(
            config["measures"]["test false positive rate"]
        )

