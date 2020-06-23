# -*- coding: utf-8 -*-

"""
Name: homogenous.py
Authors: Christian Haack, Stephan Meighen-Berger, Andrea Turcati
"""
from typing import Union, Tuple

import numpy as np  # type: ignore

import logging

from ..pdfs import construct_pdf, PDF
from ..config import config
from .population_base import Population

_log = logging.getLogger(__name__)


class HomogeneousPopulation(Population):

    def __init__(self, *args, **kwargs):
        contact_pdf = construct_pdf(
            config["population"]["social circle interactions pdf"]
        )

        self._contact_pdf = contact_pdf

    def set_contact_pdf(self, pdf: PDF):
        self._contact_pdf = construct_pdf(pdf)

    def get_contacts(
        self, rows: np.ndarray, cols: np.ndarray, return_rows=False
    ) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        """
        Get the contacts for indices in `rows`

        Parameters:
            rows np.ndarray
            return_rows: Optional[bool]
                Return the rows with non-zero interactions

        Returns:
            contact_indices: np.ndarray
            contact_strengths: np.ndarray
        """

        n_contacts = self._contact_pdf.rvs(rows.shape[0])

        # reduce by uninfectable
        n_contacts *= len(cols) / self._pop_size
        n_contacts_sym = np.asarray(np.round(n_contacts), dtype=np.int)

        all_sel_indices = self._rstate.randint(
                0, len(cols), size=np.sum(n_contacts_sym), dtype=np.int
        )
        all_sel_indices = cols[all_sel_indices]

        succesful_rows = []
        # contact_rates = []

        if len(all_sel_indices) > 0:
            all_sel_indices_split = np.split(
                all_sel_indices,
                np.cumsum(n_contacts_sym),
            )[:-1]

            for i, (row_index, sel_indices) in enumerate(
                zip(rows, all_sel_indices_split)
            ):
                if len(all_sel_indices_split[i]) == 0:
                    continue

                succesful_rows.append(
                    [row_index] * len(all_sel_indices_split[i]))

            succesful_rows = np.concatenate(succesful_rows)

            unique_indices, ind, counts = np.unique(
                all_sel_indices, return_index=True, return_counts=True
            )
            sel_indices = unique_indices

            succesful_rows = succesful_rows[ind]
            contact_rates = np.ones_like(unique_indices) * counts

        else:
            sel_indices = np.empty(0, dtype=int)
            contact_rates = np.empty(0, dtype=int)
            succesful_rows = np.empty(0, dtype=int)

        if return_rows:
            return sel_indices, contact_rates, succesful_rows
        return sel_indices, contact_rates
