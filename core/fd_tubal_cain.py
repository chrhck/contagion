"""
Name: fd_tubal_cain.py
Authors: Stephan Meighen-Berger
Creats the light spectrum pdfs.
Converts the created pdfs into a single usable
function
"""

"Imports"
from sys import exit
import numpy as np
from fd_config import config
from scipy.interpolate import UnivariateSpline

class fd_tubal_cain(object):
    """
    class: fd_tubal_cain
    Creates a single usable function from the species pdfs.
    Parameters:
        -dic pdfs:
            The species pdfs
        -obj log:
            The logger
    Returns:
        -None
    "forger of all instruments of bronze and iron"
    """
    def __init__(self, pdfs, log):
        """
        function: __init__
        Initalizes the smith Tubal-cain.
        Forges a single distribution function
        Parameters:
            -dic pdfs:
                The species pdfs
            -obj log:
                The logger
        Returns:
            -None
        """
        # Saving pdf array structure for later usage
        log.debug('Defining the species order')
        self.__keys = np.array(
            [
                key
                for key in pdfs.keys()
            ]
        )
        # This array is fixed from now on
        # The weights for this array should correspond
        # to self.__keys__
        log.debug('Constructing the pdf array')
        self.__pdf_array = np.array(
            [
                pdfs[key]
                for key in self.__keys
            ]
        )
        log.debug('Constructing a uniform population')
        self.__population_var = np.reshape(
            np.ones(len(self.__pdf_array)),
            (len(self.__pdf_array), 1)
        )

    def fd_smithing(self, population=None):
        """
        function: fd_smithing
        Forges a spline from the pdfs.
        As a standard all particles have the same population.
        This can be changed by setting population to an array filled
        with the weights for each species. The weights are orderd
        according to self.keys
        Parameters:
            -optional np.array population:
                Needs to be of shape (len(pdfs), 1)
        Returns:
            -obj spl:
                The resulting spline
        """
        if population is None:
            population = self.__population_var
        if population.shape != (len(self.__pdf_array), 1):
            exit('The shape of the population array is wrong!')
        spl = UnivariateSpline(
            config['pdf_grid'],
            np.sum(self.__pdf_array * population / np.sum(population), axis=0),
            ext=3,
            s=0,
            k=1
        )
        return spl

    @property
    def keys(self):
        """
        function: keys
        Fetches the organized keys
        Parameters:
            -None
        Returns:
            -np.array keys:
                The organized keys
        """
        return self.__keys
