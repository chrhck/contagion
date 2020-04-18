import logging
from time import time

from .state_machine import StateMachine
_log = logging.getLogger(__name__)


class StandardScenario(object):

    def __init__(
            self,
            state_machine: StateMachine,
            sim_length: int,
            *args, **kwargs):
        self._sim_length = sim_length
        self._sm = state_machine

        _log.info("There will be %d simulation steps", self._sim_length)

    def run(self):
        start = time()

        for step in range(self._sim_length):
            self._sm.tick()
            if step % (self._sim_length / 10) == 0:
                end = time()
                _log.debug("In step %d" % step)
                _log.debug(
                    "Last round of simulations took %f seconds" % (end - start)
                )
                start = time()


class SocialDistancing(StandardScenario):

    def __init__(
            self,
            state_machine: StateMachine,
            sim_length: int,
            t_start: int,
            t_stop: int,
            contact_rate_scaling: float,
            *args, **kwargs):
        super().__init__(state_machine, sim_length, *args, **kwargs)

        self._t_start = t_start
        self._t_stop = t_stop
        self._contact_rate_scaling = contact_rate_scaling

    def run(self):
        start = time()

        # old_contact_func = None
        for step in range(self._sim_length):

            if step == self._t_start:
                _log.debug("Start social distancing")

                old_rate_scale = self._sm._population.interaction_rate_scaling
                self._sm._population.interaction_rate_scaling =\
                        self._contact_rate_scaling

                """
                old_contact_func = self._sm._population.get_contacts

                def wrapped_get_contacts(
                        rows: np.ndarray,
                        cols: np.ndarray,
                        return_rows=False):

                    res = old_contact_func(rows, cols, return_rows)

                    if return_rows:
                        sel_indices, contact_rates, succesful_rows = res
                        contact_rates *= self._contact_rate_scaling
                        return sel_indices, contact_rates, succesful_rows
                    return (sel_indices,
                            contact_rates*self._contact_rate_scaling)

                self._sm._population.get_contacts = wrapped_get_contacts
                """

            if step == self._t_stop:
                # self._sm._population.get_contacts = old_contact_func
                self._sm._population.interaction_rate_scaling =\
                    old_rate_scale

            self._sm.tick()
            if step % (self._sim_length / 10) == 0:
                end = time()
                _log.debug("In step %d" % step)
                _log.debug(
                    "Last round of simulations took %f seconds" % (end - start)
                )
                start = time()
