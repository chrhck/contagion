# -*- coding: utf-8 -*-

"""
Name: population_base.py
Authors: Christian Haack, Stephan Meighen-Berger, Andrea Turcati
"""
import abc
from typing import Union, Tuple
import random
import logging

import numpy as np


from .config import config

_log = logging.getLogger(__name__)


class Population(object, metaclass=abc.ABCMeta):
    def __init__(self, *args, **kwargs):
        # Checking random state
        self._pop_size = config["population"]["population size"]
        self._rstate = config["runtime"]["random state"]
        random.seed(config["general"]["random state seed"])

    @abc.abstractmethod
    def get_contacts(
        self, contactee: np.ndarray, contacts: np.ndarray, return_rows=False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        pass
