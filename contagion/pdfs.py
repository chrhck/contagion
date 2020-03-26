"""
This module provided interfaces to PDFs and RNG
"""
import abc
from typing import Optional, Union
import scipy.stats  # type: ignore
import numpy as np  # type: ignore

from .config import config


class Probability(object, metaclass=abc.ABCMeta):
    """Interface for probabilities"""

    @abc.abstractmethod
    def __call__(self, values: np.ndarray):
        """Return the probabilities"""
        pass


class PDF(object, metaclass=abc.ABCMeta):
    """Interface for PDFs"""

    @abc.abstractmethod
    def rvs(self, num: int) -> np.ndarray:
        """Return rvs sampled from the PDF"""
        pass


class ScipyPDF(PDF, metaclass=abc.ABCMeta):

    def rvs(self, num: int, dtype: Optional[type] = None) -> np.ndarray:
        rvs = self._pdf.rvs(
            size=num, random_state=config['random state'])

        if dtype is not None:
            rvs = np.asarray(rvs, dtype=dtype)
        return rvs


class TruncatedNormal(ScipyPDF):
    """Truncated normal distribution

    Parameters:
        lower: Union[float, np.ndarray]
            Lower bound (can be -inf)
        upper: Union[float, np.ndarray]
            Upper bound (can be +inf)
        mean: Union[float, np.ndarray]
        sd: Union[float, np.ndarray]
    """

    def __init__(
            self,
            lower: Union[float, np.ndarray],
            upper: Union[float, np.ndarray],
            mean: Union[float, np.ndarray],
            sd: Union[float, np.ndarray]) -> None:
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
    """Beta distribution

    Parameters:
        mean: Union[float, np.ndarray]
            Mean of the distribution
        sd: Union[float, np.ndarray]
            Standard deviation (has to be smaller than sqrt(mean(1-mean)))
    """

    def __init__(self, mean, sd):
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
    Uniform PDF

    Parameters:
        lower: Union[float, np.ndarray]
            Lower bound
        upper: Union[float, np.ndarray]
            Upper bound
    """

    def __init__(
            self,
            lower: Union[float, np.ndarray],
            upper: Union[float, np.ndarray]
            ) -> None:
        self._lower = lower
        self._upper = upper
        self._pdf = scipy.stats.uniform(self._lower, self._upper)


class NormalizedProbability(Probability):
    """Normalizes a range to a probability in [0, 1]"""

    def __init__(self, lower: int, upper: int) -> None:
        super().__init__()

        self._lower = lower
        self._upper = upper
        self._interval_length = self._upper - self._lower

    def __call__(self, values: np.ndarray):
        """Normalize the values to [0, 1]"""

        values = np.atleast_1d(values)
        if ~np.all((values <= self._upper) &
                   (values >= self._lower)):
            raise ValueError("Not all values in range")

        return (values - self._lower) / self._interval_length
