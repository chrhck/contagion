"""
yom.py
Authors: Stephan Meighen-Berger
Class to manage the pulse length of the
organisms
"""

"Imports"
from fd_config import config
from sys import exit

class fd_yom(object):
    """
    class: fd_yom
    Manages the pulse length of the creatures
    Parameters:
        -np.array pulses:
            The pulses calculated
        -obj. log:
            The logger
    Returns:
        -None
    "Making the best use of the time,
     because the days are evil.
     Therefore do not be foolish,
     but understand what the will of the Lord is."
    """

    def __init__(self, pulse, log):
        """
        function: __init__
        Initializes the class
        Parameters:
            -np.array pulses
                The calculated pulses
            -obj. log:
                The logger
        Returns:
            -None
        """
        log.debug("Applying pulse shapes")
        if config["pulse shape"] == "uniform":
            log.debug("Using uniform pulse shapes")
            self.__pulse_res = pulse
        else:
            log.error("Pulse shape not implemented!")
            exit("Check the config file and the pulse shape")

    @property
    def shaped_pulse(self):
        """
        function: shaped_pulse
        Pulse after applying emission shapes
        Parameters:
            -None
        Returns:
            -np.array the reshaped pulses
        """
        return self.__pulse_res
