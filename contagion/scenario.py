import logging
from typing import List
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
            stop = self._sm.tick()
            if stop:
                _log.debug("Early stopping at %d", step)
                break
            if step % (self._sim_length / 10) == 0:
                end = time()
                _log.debug("In step %d" % step)
                _log.debug(
                    "Last round of simulations took %f seconds" % (end - start)
                )
                start = time()


class LateMeasures(StandardScenario):
    def __init__(
            self,
            state_machine: StateMachine,
            sim_length: int,
            start_measures_inf_frac: float,
            *args, **kwargs):
        super().__init__(state_machine, sim_length, *args, **kwargs)

        self._start_measures_inf_frac = start_measures_inf_frac
        self._measures_active = False

    def run(self):
        start = time()

        for step in range(self._sim_length):
            # print("day: ", step)
            inf_frac = (
                self._sm._data["is_infected"].sum() /
                self._sm._data.field_len
                )
            if (inf_frac > self._start_measures_inf_frac or
                 self._measures_active):
                self._sm._measures.measures_active = True
                self._measures_active = True
            else:
                self._sm._measures.measures_active = False

            stop = self._sm.tick()
            if stop:
                _log.debug("Early stopping at %d", step)
                break
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
            t_steps: List[int],
            contact_rate_scalings: List[int],
            *args, **kwargs):
        super().__init__(state_machine, sim_length, *args, **kwargs)

        self._t_steps = t_steps[::-1]
        self._contact_rate_scalings = contact_rate_scalings[::-1]

    def run(self):
        start = time()

        next_change = self._t_steps.pop()
        next_scaling = self._contact_rate_scalings.pop()

        # old_contact_func = None
        for step in range(self._sim_length):
            if step == next_change:
                _log.debug("New social distancing step")
                self._sm._population.interaction_rate_scaling = next_scaling
                if self._t_steps:
                    next_change = self._t_steps.pop()
                    next_scaling = self._contact_rate_scalings.pop()
                else:
                    next_change = None

            self._sm.tick()
            if step % (self._sim_length / 10) == 0:
                end = time()
                _log.debug("In step %d" % step)
                _log.debug(
                    "Last round of simulations took %f seconds" % (end - start)
                )
                start = time()
