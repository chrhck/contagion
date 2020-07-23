"""
This module provided interfaces to PDFs and RNG
"""
import abc
import logging
from typing import Optional, Union, Dict, Any
import scipy.stats  # type: ignore
import numpy as np  # type: ignore

from .config import config

_log = logging.getLogger(__name__)


class Probability(object, metaclass=abc.ABCMeta):
    """
    Abstract class to interface to the probabilities

    Subclasses have to implement `__call__`
    """

    @abc.abstractmethod
    def __call__(self, values: np.ndarray) -> np.ndarray:
        """
        Calculate the probabilites for `values`

        Parameters:
            values: np.narray

        """
        pass


class PDF(object, metaclass=abc.ABCMeta):
    """
    Metaclass for pdfs.

    Subclasses have to implement `rvs`
    """

    @abc.abstractmethod
    def rvs(self, num: int, dtype: Optional[type] = None) -> np.ndarray:
        """
        Random variate sampling

        Parameters:
            num: int
                Number of samples to draw
            dtype: Optional[type]
                dtype of the returned samples
        Returns:
            rvs: np.ndarray
                The drawn samples
        """
        pass


class Delta(PDF):
    """
    Delta distribution

    Returns a fixed value.

    Parameters:
        mean: Union[float, np.ndarray]
    """
    def __init__(
            self,
            mean: Union[float, np.ndarray],
            ):
        self._mean = mean

    def rvs(self, num: int, dtype: Optional[type] = None) -> np.ndarray:
        return np.ones(num, dtype=dtype)*self._mean


class ScipyPDF(PDF, metaclass=abc.ABCMeta):
    """
    Interface to scipy distributions

    Parameters:
        lower, upper: float
            Lower and upper boundaries for returned samples
    """

    def __init__(self, lower: float, upper: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lower = lower
        self._upper = upper

    def rvs(self, num: int, dtype: Optional[type] = None) -> np.ndarray:
        if num == 0:
            return np.zeros(0)
        samples = []
        s = 0

        while s < num:
            sample = self._rvs(num, dtype)
            select = sample[
                np.logical_and(sample >= self._lower, sample <= self._upper)]
            samples.append(select)
            s += len(select)
        return np.concatenate(samples)[:num]

    def _rvs(self, num: int, dtype: Optional[type] = None) -> np.ndarray:
        """
        Call the underlying scipy distributon

        Parameters:
            num: int
                Number of samples
            dtype: Optional[type]
                Type of the output
        Returns:
            rvs: np.array
                The drawn samples
        """
        rvs = self._pdf.rvs(
            size=num, random_state=config["runtime"]["random state"])

        if dtype is not None:
            rvs = np.asarray(rvs, dtype=dtype)
        return rvs

    def pdf(self, points: Union[float, np.ndarray],
            dtype: Optional[type] = None) -> np.ndarray:
        """
        Calculates the pdf at given values
        Parameters:
            points:  Union[float, np.ndarray]
            dtype: Optional[type]
                dtype of the returned values
        Returns:
            pdf: np.array
        """
        pdf = self._pdf.pdf(
            points
        )

        if dtype is not None:
            pdf = np.asarray(pdf, dtype=dtype)
        return np.nan_to_num(pdf)


class TruncatedNormal(ScipyPDF):
    """
    Class for the truncated normal distributon

    Parameters:
        mean: Union[float, np.ndarray]
        sd: Union[float, np.ndarray]
        lower, upper: float
            Lower and upper boundaries for returned samples
    """

    def __init__(
            self,
            mean: Union[float, np.ndarray],
            sd: Union[float, np.ndarray],
            lower=-np.inf,
            upper=np.inf) -> None:
        super().__init__(lower, upper)
        # Other cases aren't used
        self._mean = mean
        self._sd = sd

        # Calculate parameter a and b.
        # See:
        #  https://docs.scipy.org/doc/scipy/
        #  reference/generated/scipy.stats.truncnorm.html
        self._a = (self._lower - self._mean) / self._sd
        self._b = (self._upper - self._mean) / self._sd

        self._pdf = scipy.stats.truncnorm(
            self._a, self._b, loc=self._mean, scale=self._sd)


class Beta(ScipyPDF):
    """
    Class for the beta distribution
    Parameters:
        mean: Union[float, np.ndarray]
            Mean of the distribution
        sd: Union[float, np.ndarray]
             Standard deviation (has to be smaller than sqrt(mean(1-mean)))
        lower, upper: float
            Lower and upper boundaries for returned samples
    """

    def __init__(self, mean, sd, lower=-np.inf, upper=np.inf):
        super().__init__(lower, upper)
        self._mean = np.atleast_1d(mean)
        self._sd = np.atleast_1d(sd)

        varmax = self._mean * (1-self._mean)
        if np.any(self._sd > np.sqrt(varmax)):
            raise ValueError("SD has to be < sqrt(mean(1-mean))")

        self._alpha = self._mean * (varmax / self._sd**2 - 1)
        self._beta = self._alpha * (1/self._mean-1)

        self._pdf = scipy.stats.beta(self._alpha, self._beta)


class Pareto(ScipyPDF):
    """
    Class for the Pareto distribution
    Parameters:
        x_min: Union[float, np.ndarray]
            Min value (mode)
        index: Union[float, np.ndarray]
            Powerlaw index
        lower, upper: float
            Lower and upper boundaries for returned samples
    """

    def __init__(self, x_min, index, lower=-np.inf, upper=np.inf):
        super().__init__(x_min, upper)
        self._x_min = np.atleast_1d(x_min)
        self._index = np.atleast_1d(index)

        self._pdf = scipy.stats.pareto(self._index, scale=self._x_min)


class Uniform(ScipyPDF):
    """
    Class for the uniform distribution
    Parameters:
        lower: Union[float, np.ndarray]
            Lower bound
        upper: Union[float, np.ndarray]
            Upper bound
    """

    def __init__(self, lower: float, upper: float) -> None:
        self._lower = lower
        self._upper = upper
        self._pdf = scipy.stats.uniform(self._lower, self._upper)


class Gamma(ScipyPDF):
    """
    Class for the gamma distribution
    Parameters:
        mean: Union[float, np.array]:
            Mean value
        std: Union[float, np.array]
            Standard deviation
        max_val: Optional[float]
            Maximum value of the pdf. Adjusts the normalization such
            that the mode is equal to max_val
        scaling: Optional[float]
            Scale the pdf by this value.
        as_dtype: Optional[type]
            dtype of the rvs. Overwrites dtype specified in call to `rvs``.
        lower, upper: float
            Lower and upper boundaries for returned samples
    """

    def __init__(
            self,
            mean: Union[float, np.ndarray],
            sd: Union[float, np.ndarray],
            max_val: Optional[float] = None,
            scaling: Optional[float] = None,
            as_dtype: Optional[type] = np.float,
            lower: Optional[float] = -np.inf,
            upper: Optional[float] = np.inf) -> None:

        super().__init__(lower, upper)
        if max_val is not None and scaling is not None:
            raise ValueError("Cannot set both scaling and max_val")
        self._mean = mean
        self._sd = sd
        self._beta = self._mean / self._sd**2.
        self._alpha = self._mean**2. / self._sd**2.
        # scipy parameters
        self._shape = self._alpha
        self._scale = 1. / self._beta
        self._mode = (self._alpha-1) / self._beta
        self._max_val = max_val
        self._scaling = scaling
        self._as_dtype = as_dtype
        self._pdf = scipy.stats.gamma(
            self._shape,
            scale=self._scale
        )

        self._val_at_mode = self._pdf.pdf(self._mode)

    def pdf(self, points: Union[float, np.ndarray],
            dtype: Optional[type] = None) -> np.ndarray:

        pdf_vals = super().pdf(points, dtype)

        if self._max_val is not None:
            pdf_vals = pdf_vals / self._val_at_mode * self._max_val

        elif self._scaling is not None:
            pdf_vals = pdf_vals * self._scaling

        return pdf_vals

    def _rvs(self, num: int, dtype: Optional[type] = None) -> np.ndarray:
        """
        Call the underlying scipy distributon

        Parameters:
            num: int
                Number of samples
            dtype: Optional[type]
                Type of the output
        Returns:
            rvs: np.array
                The drawn samples
        """
        # scipy rvs is slow
        rstate = config["runtime"]["random state"]

        # This is much faster than the scipy implementation
        rvs = rstate.standard_gamma(self._shape, size=num) * self._scale

        return rvs.astype(self._as_dtype)


# class NormalizedProbability(Probability):
#     """
#     Uniform probability on an interval.
#
#     The probability is calculated as 1/(upper - lower)
#
#     Parameters:
#         lower: int
#             Lower bound
#         -int upper:
#             The upper bound
#     Returns:
#         -None
#     """
#
#     def __init__(self, lower: int, upper: int) -> None:
#         """
#         function: __init__
#         Initializes the normalization class
#         Parameters:
#             -int lower:
#                 The lower bound
#             -int upper:
#                 The upper bound
#         Returns:
#             -None
#         """
#         super().__init__()
#
#         self._lower = lower
#         self._upper = upper
#         self._interval_length = self._upper - self._lower
#
#     def __call__(self, values: np.ndarray):
#         """
#         function: __call__
#         Call handling
#         Parameters:
#             -np.array values:
#                 The values to flatten
#         Returns:
#             -None
#         """
#
#         values = np.atleast_1d(values)
#         if ~np.all((values <= self._upper) &
#                    (values >= self._lower)):
#             raise ValueError("Not all values in range")
#
#         return (values - self._lower) / self._interval_length


def construct_pdf(conf_dict: Dict[str, Any]) -> PDF:
    """
    Convenience function to create a PDF from a config dict

    Parameters:
        conf_dict: Dict[str, Any]
            The dict should contain a `class` key with the name of the
            PDF to instantiate. Any further keys will be passed as kwargs
    """
    try:
        conf_dict = dict(conf_dict)
        class_name = conf_dict.pop("class")
        pdf_class = globals()[class_name]
        pdf = pdf_class(**conf_dict)
    except KeyError:
        _log.error("Unknown pdf class: %s", class_name)
        raise KeyError(("Unknown pdf class: %s".format(class_name)))
    return pdf
