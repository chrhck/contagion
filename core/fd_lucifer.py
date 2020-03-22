"""
Name: fd_lucifer.py
Authors: Stephan Meighen-Berger
Propagates the light through the material
"""

"Imports"
import numpy as np


class fd_lucifer(object):
    """
    class: fd_lucifer
    Propagates light through the material
    Parameters:
        -np.array light_yield:
            The light yield
        -np.array distances:
            The observation distances
        -obj. log:
            The logger
    Returns:
        -None
    "How you have fallen from heaven, morning star,
     son of the dawn! You have been cast down to the earth,
     you who once laid low the nations!"
    """

    def __init__(self, light_yields, distances, log):
        """
        function: __init__
        Initializes the class
        Parameters:
            -np.array light_yield:
                The light yield
            -np.array distances:
                The observation distances
            -obj. log:
                The logger
        Returns:
            -None
        """
        log.debug('Calculating attenuated light')
        self.__light_yields = np.array([
            light_yields *self.__attenuation(dist)
            for dist in distances
        ])

    @property
    def yields(self):
        """
        function: yields
        Getter function for attenuated light
        Parameters:
            -None
        Returns:
            -dic life:
                The created organisms
        """
        return self.__light_yields

    def __attenuation(self, distance):
        """
        function: __attenuation
        Attenuates the light according to the observation
        distance
        Parameters:
            -float distance:
                The observation distance
        Returns:
            -float res:
                The attenuation factor
        """
        return np.exp(- distance / 6.9)
