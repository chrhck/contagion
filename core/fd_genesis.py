"""
Name: fd_genesis.py
Authors: Stephan Meighen-Berger
Creats the light spectrum pdf.
This is used to fit the data.
"""

"Imports"
from sys import exit
import numpy as np
from fd_config import config
from scipy.stats import gamma
from scipy.signal import peak_widths
from scipy.optimize import root

class fd_genesis(object):
    """
    class: fd_genesis
    Creates the light distributions from
    organisms.
    Parameters:
        -dic life:
            The organisms created
        -obj log:
            The logger
    Returns:
        -None
    "And God saw the light, that it was good:
     and God divided the light from the darkness"
    """
    def __init__(self, life, log):
        """
        function: __init__
        initializes genesis.
        Parameters:
            -dic life:
                The organisms created
            -obj log:
                The logger
        Returns:
            -None
        """
        self.__log = log
        # These points are used in solving
        self.__x = config['pdf_grid']
        self.__pdfs = {}
        if config['pdf'] == 'gamma':
            self.__log.debug('Genesis of Gamma distributions')
            for key in life.keys():
                for idspecies, _ in enumerate(life[key][0]):
                    param = self.__forming(
                        [life[key][1][idspecies], life[key][2][idspecies]]
                    )
                    self.__pdfs[life[key][0][idspecies]] = (
                        gamma.pdf(self.__x, param[0], scale=param[1])
                    )
        else:
            self.__log.error('Distribution unknown!')
            exit()
    
    def __forming(self, species):
        """
        function: __forming
        Creates the light pdf for the species.
        Parameters:
            -np.array species:
                The species parameters
        Returns:
            -nparray pdf:
                The constructed pdf
        """
        # The mean
        mean = species[0]
        # The FWHM
        fwhm = species[1]
        # The equation to solve
        def equation(k):
            scale = mean / k
            signal = gamma.pdf(self.__x, k, scale=scale)
            peaks = signal.argmax()
            curr_fwhm = peak_widths(signal, [peaks], rel_height=0.5)
            width = (curr_fwhm[-1] - curr_fwhm[-2])[0]
            res = width - fwhm
            return res
        res = root(equation, 100.).x[0]
        return np.array([res, mean / res])

    @property
    def pdfs(self):
        """
        function: pdfs
        Getter function for the pdfs
        Parameters:
            -None
        Returns:
            -dic pdfs:
                The light emission pdfs for the species
        """
        return self.__pdfs
