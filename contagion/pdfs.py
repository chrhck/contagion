"""
This module provided interfaces to PDFs and RNG
"""
import abc
from typing import Optional, Union
import scipy.stats  # type: ignore
import numpy as np  # type: ignore

from .config import config

# TODO: Upate formatting
class Probability(object, metaclass=abc.ABCMeta):
    """
    class: Probability
    Abstract class to interface to the probabilities
    Parameters:
        -class pdf:
            The probability distribution class
    Returns:
        -None
    """

    @abc.abstractmethod
    def __call__(self, values: np.ndarray):
        """
        function: __call__
        Calls the normalized pdfs
        Parameters:
            -np.array values:
                The values to flatten
        Returns:
            -None
        """
        pass


class PDF(object, metaclass=abc.ABCMeta):
    """
    class: PDF
    Interface class for the pdfs.
    One layer inbetween to allow different
    pdf packages such as scipy and numpy
    Parameters:
        -class pdf_interface:
            Interface class to the pdf
            classes
    Returns:
        -None
    """

    @abc.abstractmethod
    def rvs(self, num: int) -> np.ndarray:
        """
        function: rvs
        Random variate sampling, filled
        with the subclasses definition
        Parameters:
            -int num:
                Number of samples to draw
        Returns:
            -np.array rvs:
                The drawn samples
        """
        pass


class ScipyPDF(PDF, metaclass=abc.ABCMeta):
    """
    class: ScipyPDF
    Class pdf classes are inheriting from.
    Deals with rvs currently
    Parameters:
        -None
    Returns:
        -None
    """
    def rvs(self, num: int, dtype: Optional[type] = None) -> np.ndarray:
        """
        function: rvs
        Calculates the random variates
        Parameters:
            -int num:
                The number of samples
            -optional dtype:
                Type of the output
        Returns:
            -np.array rvs:
                The drawn samples
        """
        rvs = self._pdf.rvs(
            size=num, random_state=config['random state'])

        if dtype is not None:
            rvs = np.asarray(rvs, dtype=dtype)
        return rvs


class TruncatedNormal(ScipyPDF):
    """
    class: TuncatedNormal
    Class for the truncated normal distributon
    Parameters:
        lower: Union[float, np.ndarray]
            Lower bound (can be -inf)
        upper: Union[float, np.ndarray]
            Upper bound (can be +inf)
        mean: Union[float, np.ndarray]
        sd: Union[float, np.ndarray]
    Returns:
        -None
    """

    def __init__(
            self,
            lower: Union[float, np.ndarray],
            upper: Union[float, np.ndarray],
            mean: Union[float, np.ndarray],
            sd: Union[float, np.ndarray]) -> None:
        """
        function: __init__
        Initializes the TruncatedNormal class
        Parameters:
            lower: Union[float, np.ndarray]
                Lower bound (can be -inf)
            upper: Union[float, np.ndarray]
                Upper bound (can be +inf)
            mean: Union[float, np.ndarray]
            sd: Union[float, np.ndarray]
        Returns:
            -None
        """
        super().__init__()
        self._lower = lower
        self._upper = upper
        self._mean = mean
        self._sd = sd

        # Calculate parameter a and b.
        # See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
        self._a = (self._lower - self._mean) / self._sd
        self._b = (self._upper - self._mean) / self._sd

        self._pdf = scipy.stats.truncnorm(
            self._a, self._b, loc=self._mean, scale=self._sd)


class Beta(ScipyPDF):
    """
    class: Beta
    Class for the beta distribution
    Parameters:
        -mean: Union[float, np.ndarray]
            Mean of the distribution
        -sd: Union[float, np.ndarray]
             Standard deviation (has to be smaller than sqrt(mean(1-mean)))
    Returns:
        -None
    """

    def __init__(self, mean, sd):
        """
        function: __init__
        Initializes the Beta class
        Parameters:
            -mean: Union[float, np.ndarray]
                Mean of the distribution
            -sd: Union[float, np.ndarray]
                Standard deviation (has to be smaller than sqrt(mean(1-mean)))
        Returns:
            -None
        """
        self._mean = np.atleast_1d(mean)
        self._sd = np.atleast_1d(sd)

        varmax = self._mean * (1-self._mean)
        if np.any(self._sd > np.sqrt(varmax)):
            raise ValueError("SD has to be < sqrt(mean(1-mean))")

        self._alpha = self._mean * (varmax / self._sd**2 - 1)
        self._beta = self._alpha * (1/self._mean-1)

        self._pdf = scipy.stats.beta(self._alpha, self._beta)


class Uniform(ScipyPDF):
    """
    class: Uniform
    Class for the uniform distribution
    Parameters:
        lower: Union[float, np.ndarray]
            Lower bound
        upper: Union[float, np.ndarray]
            Upper bound
    Returns:
        -None
    """

    def __init__(
            self,
            lower: Union[float, np.ndarray],
            upper: Union[float, np.ndarray]
            ) -> None:
        """
        function: __init__
        Initializes the Uniform class
        Parameters:
            -lower: Union[float, np.ndarray]
                    Lower bound
            -upper: Union[float, np.ndarray]
                    Upper bound
        Returns:
            -None
        """
        self._lower = lower
        self._upper = upper
        self._pdf = scipy.stats.uniform(self._lower, self._upper)


class NormalizedProbability(Probability):
    """
    class: NormalizedProbability
    Class to normalize the pdf to the interval [0,1]
    Parameters:
        -int lower:
            The lower bound
        -int upper:
            The upper bound
    Returns:
        -None
    """

    def __init__(self, lower: int, upper: int) -> None:
        """
        function: __init__
        Initializes the normalization class
        Parameters:
            -int lower:
                The lower bound
            -int upper:
                The upper bound
        Returns:
            -None
        """
        super().__init__()

        self._lower = lower
        self._upper = upper
        self._interval_length = self._upper - self._lower

    def __call__(self, values: np.ndarray):
        """
        function: __call__
        Call handling
        Parameters:
            -np.array values:
                The values to flatten
        Returns:
            -None
        """

        values = np.atleast_1d(values)
        if ~np.all((values <= self._upper) &
                   (values >= self._lower)):
            raise ValueError("Not all values in range")

        return (values - self._lower) / self._interval_length