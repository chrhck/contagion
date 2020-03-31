from __future__ import annotations
import abc
from collections import defaultdict
import functools
import logging
from typing import Callable, Union, List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from scipy import sparse

from .infection import Infection
from .pdfs import PDF


_log = logging.getLogger(__name__)

DEBUG = False


class DataDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        field_len = None
        for key, val in self.items():
            if field_len is not None and len(val) != field_len:
                raise RuntimeError("Not all fields are of same length")
            field_len = len(val)

        self._field_len = field_len

    @property
    def field_len(self):
        return self._field_len


class Condition(object):
    """
    Convenience class for storing references to conditions
    """
    def __init__(self, condition: Callable):
        self._condition = condition

    @classmethod
    def from_state(cls, state: _State):
        return cls(state)

    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, val):
        self._condition = val

    def __call__(self, data: DataDict):
        """Evaluate condition on DataFrame"""
        return self.condition(data)

    def __and__(self, other: TCondition):
        def new_condition(data: DataDict):
            cond = unify_condition(other, data)

            return self(data) & cond
        return Condition(new_condition)


TCondition = Union["_State", np.ndarray, Condition]


def unify_condition(condition: TCondition, data: DataDict) -> np.ndarray:
    if isinstance(condition, (_State, Condition)):
        cond = condition(data)
    elif isinstance(condition, np.ndarray):
        cond = condition
    elif condition is None:
        cond = np.ones(data.field_len, dtype=np.bool)
    else:
        raise ValueError("Unsupported type: ", type(condition))
    return cond


class ConditionalMixin(object):

    def __init__(self, condition, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._condition = condition

    def unify_condition(self, data: DataDict):
        return unify_condition(self._condition, data)


class _State(object, metaclass=abc.ABCMeta):
    """Interface for all States"""

    def __init__(
            self,
            state_getter: Callable,
            state_value_getter: Callable,
            name: str,
            data_field: str,
            state_change: Callable,
            *args, **kwargs):

        self._state_getter = state_getter
        self._state_value_getter = state_value_getter
        self._name = name
        self._state_change = state_change
        self._data_field = data_field

    def __call__(self, data: DataDict):
        """
        Returns the state

        Parameters:
            df: pd.DataFrame

        Returns:
            pd.Series
        """
        return self._state_getter(data[self._data_field])

    def get_state_value(self, data: DataDict):
        return self._state_value_getter(data[self._data_field])

    def __invert__(self):
        """
        Return a state with inverted condition
        """
        def inverted_condition(arr: np.ndarray):
            return ~(self._state_getter(arr))

        return type(self)(
            inverted_condition,
            self._state_value_getter,
            "inverted_" + self.name,
            self._data_field,
            self._state_change)

    def change_state(
            self,
            data: DataDict,
            state: np.ndarray,
            condition: TCondition = None,
            ):
        """Changes the state in the DataFrame"""
        # Check which is currently in this state

        self_cond = self(data)
        cond = unify_condition(condition, data)
        self._state_change(data, state, cond & self_cond)

    @property
    def name(self):
        return self._name


class BooleanState(_State):
    """
    Determine the state for every row in a DataFrame

    When called, an object of this class returns a boolean series
    that shows whether each row is in the state determined by
    the state_condition

    Parameters:
        state_condition: Callable
            This function should take the dataframe as only argument
            and return a boolean series with the same index as the dataframe
        name: str
            The state's name

    """

    @classmethod
    def from_boolean(cls, name: str):
        def get_state(arr: np.ndarray):
            return arr

        def state_change(
                data: DataDict,
                state: np.ndarray,
                condition: np.ndarray):

            # TODO: maybe offload application of condition to state here?
            data[name][condition] = state

        return cls(get_state, get_state, name, name, state_change)


class FloatState(_State):

    @classmethod
    def from_timer(cls, name: str):
        def get_state(arr: np.ndarray):
            return arr > 0

        def get_state_value(arr: np.ndarray):
            return arr

        def state_change(data: DataDict, state: np.ndarray, condition):
            data[name][condition] = state

        return cls(get_state, get_state_value, name, name, state_change)


def log_call(func):

    if DEBUG:
        @functools.wraps(func)
        def log_wrapper(self, data):
            _log.debug("Performing %s", self.name)
            if hasattr(self, "_condition"):
                cond = self.unify_condition(data)
                _log.debug("Condition is: %s", cond)
            df_before = pd.DataFrame(data, copy=True)
            retval = func(self, data)
            df_after = pd.DataFrame(data, copy=True)

            diff = df_before.astype("float") - df_after.astype("float")

            diff_rows = diff.loc[diff.any(axis=1), :]
            diff_cols = diff_rows.loc[:, diff_rows.any(axis=0)]
            _log.debug(
                "Dataframe diff: %s", diff_cols
                )

            return retval
        return log_wrapper
    else:
        return func


class _Transition(object, metaclass=abc.ABCMeta):
    """Interface for Transitions"""

    def __init__(self, name, *args, **kwargs):
        self._name = name

    @abc.abstractmethod
    def __call__(self, data: DataDict):
        pass

    @property
    def name(self):
        return self._name


class Transition(_Transition):

    def __init__(
            self,
            name: str,
            state_a: _State,
            state_b: _State,
            *args, **kwargs):

        super().__init__(name, *args, **kwargs)
        self._state_a = state_a
        self._state_b = state_b

    @log_call
    def __call__(self, data: DataDict):
        """
        Perform the transition.

        All rows in `df` which where previusly in state A are transitioned
        to state B.

        Parameters:
            arr: DataDict
        """

        # Invert state B to select all rows which are _not_ in state B
        # Use state A as condition so that only rows are activated which
        # where in state A

        (~self._state_b).change_state(data, True, self._state_a(data))
        self._state_a.change_state(data, False)


class ChangeStateConditionalTransition(_Transition, ConditionalMixin):
    def __init__(
            self,
            name: str,
            state_a: Union[_State, Tuple[_State, bool]],
            condition: TCondition,
            *args, **kwargs,
           ):
        _Transition.__init__(self, name, *args, **kwargs)
        ConditionalMixin.__init__(self, condition, *args, **kwargs)

        if isinstance(state_a, tuple):
            self._state_a = state_a[0]
            self._state_a_val = state_a[1]
        else:
            self._state_a = state_a
            self._state_a_val = True

    @log_call
    def __call__(
            self,
            data: DataDict):
        cond = self.unify_condition(data)
        self._state_a.change_state(
            data, self._state_a_val, cond)

        return cond


class ConditionalTransition(_Transition, ConditionalMixin):
    def __init__(
            self,
            name: str,
            state_a: _State,
            state_b: _State,
            condition: TCondition,
            *args, **kwargs,
           ):

        _Transition.__init__(self, name, *args, **kwargs)
        ConditionalMixin.__init__(self, condition, *args, **kwargs)
        self._state_a = state_a
        self._state_b = state_b

    @log_call
    def __call__(
            self,
            data: DataDict):

        cond = self.unify_condition(data)

        (~self._state_b).change_state(data, True, cond & self._state_a(data))
        self._state_a.change_state(data, False, cond)

        return cond


class DecreaseTimerTransition(_Transition, ConditionalMixin):
    def __init__(
            self,
            name: str,
            state_a: FloatState,
            condition: TCondition,
            *args, **kwargs
           ):
        _Transition.__init__(self, name, *args, **kwargs)
        ConditionalMixin.__init__(self, condition, *args, **kwargs)
        self._state_a = state_a

    @log_call
    def __call__(
            self,
            data: DataDict):

        cond = unify_condition(self._condition, data)
        state_condition = self._state_a(data)

        cur_state = self._state_a.get_state_value(data)[
            cond & state_condition]
        self._state_a.change_state(data, cur_state-1, cond)


class InitializeTimerTransition(_Transition, ConditionalMixin):
    def __init__(
            self,
            name: str,
            state_a: FloatState,
            initialization_pdf: PDF,
            condition: TCondition,
            *args,
            **kwargs
           ):

        _Transition.__init__(self, name, *args, **kwargs)
        ConditionalMixin.__init__(self, condition)
        self._state_a = state_a
        self._initialization_pdf = initialization_pdf

    @log_call
    def __call__(
            self,
            data: DataDict):

        cond = self.unify_condition(data)

        zero_rows = (~self._state_a(data)) & cond
        num_zero_rows = zero_rows.sum(axis=0)

        initial_vals = self._initialization_pdf.rvs(num_zero_rows)
        (~self._state_a).change_state(
            data, initial_vals, cond)


class MultiStateConditionalTransition(_Transition, ConditionalMixin):

    _states_b: List[_State]
    _states_b_vals: List[bool]

    def __init__(
            self,
            name: str,
            state_a: Union[_State, Tuple[_State, bool]],
            states_b: List[Union[_State, Tuple[_State, bool]]],
            condition: TCondition,
            *args, **kwargs):

        _Transition.__init__(self, name, *args, **kwargs)
        ConditionalMixin.__init__(self, condition, *args, **kwargs)

        if isinstance(state_a, tuple):
            self._state_a = state_a[0]
            self._state_a_val = state_a[1]
        else:
            self._state_a = state_a
            self._state_a_val = False

        self._states_b = []
        self._states_b_vals = []
        for state in states_b:
            if isinstance(state, tuple):
                self._states_b.append(state[0])
                self._states_b_vals.append(state[1])
            else:
                self._states_b.append(state)
                self._states_b_vals.append(True)

    @log_call
    def __call__(self, data: DataDict):

        cond = self.unify_condition(data)

        is_in_state_a = self._state_a(data)
        for state, val in zip(self._states_b, self._states_b_vals):
            if DEBUG:
                _log.debug("Changing state %s to %s", (~state).name, val)
                _log.debug("State was: %s", (~state)(data)[cond & is_in_state_a])
            (~state).change_state(data, val, cond & is_in_state_a)
            if DEBUG:
                _log.debug("State is: %s", (~state)(data)[cond & is_in_state_a])

        self._state_a.change_state(data, self._state_a_val, cond)

        return cond


class StatCollector(object, metaclass=abc.ABCMeta):
    _statistics: Dict[str, List[float]]

    def __init__(self, data_fields):
        self._data_fields = data_fields
        self._statistics = defaultdict(list)

    def __call__(self, data: DataDict):
        for field in self._data_fields:
            self._statistics[field].append(data[field].sum())

    @property
    def statistics(self):
        return self._statistics


class StateMachine(object, metaclass=abc.ABCMeta):

    def __init__(
            self,
            df: pd.DataFrame,
            stat_colletor: Optional[StatCollector],
            *args, **kwargs):
        self._data = DataDict({key: df[key].values for key in df.columns})
        self._stat_collector = stat_colletor

    @abc.abstractmethod
    def transitions(self) -> List[_Transition]:
        pass

    @abc.abstractmethod
    def states(self) -> List[_State]:
        pass

    def tick(self):
        for transition in self.transitions:
            transition(self._data)
        if self._stat_collector is not None:
            self._stat_collector(self._data)

    @property
    def statistics(self):
        return self._stat_collector.statistics


class ContagionStateMachine(StateMachine):

    _states: Dict[str, _State]
    _statistics: Dict[str, List[float]]

    def __init__(
            self,
            df: pd.DataFrame,
            stat_colletor: Optional[StatCollector],
            interactions: sparse.spmatrix,
            infection: Infection,
            intensity_pdf: PDF,
            rstate: np.random.RandomState,
            *args, **kwargs):
        super().__init__(df, stat_colletor, *args, **kwargs)

        self._interactions = interactions.tocsr()
        self._rstate = rstate
        self._infection = infection
        self._intensity_pdf = intensity_pdf
        self._statistics = defaultdict(list)

        boolean_state_names = [
            "is_infected", "is_new_infected", "is_dead", "is_removed",
            "is_infectious", "is_new_infectious", "is_hospitalized",
            "is_new_hospitalized", "is_recovering", "is_new_recovering",
            "is_incubation", "is_new_incubation",
            "is_recovered", "will_be_hospitalized", "will_die"]

        boolean_states = {name: BooleanState.from_boolean(name)
                                 for name in boolean_state_names}

        timer_state_names = [
            "incubation_duration", "hospitalization_duration", "recovery_time",
            "time_until_hospitalization", "infectious_duration",
            "time_until_death"]

        timer_states = {name: FloatState.from_timer(name)
                        for name in timer_state_names}

        self._states = {}
        self._states.update(boolean_states)
        self._states.update(timer_states)

        infected_condition = Condition(self.__get_new_infections)

        is_infectable = (
                infected_condition & (~boolean_states["is_removed"])
            )

        will_be_hospitalized_cond = Condition(self.__will_be_hospitalized)
        will_die_cond = Condition(self.__will_die)
        normal_recovery_condition = (
            Condition.from_state(~(timer_states["infectious_duration"])) &
            Condition.from_state(~(boolean_states["is_hospitalized"]))
            )

        self._transitions = [
            MultiStateConditionalTransition(
                "healthy_incubation",
                ~boolean_states["is_infected"],
                [boolean_states["is_incubation"],
                 boolean_states["is_new_incubation"],
                 boolean_states["is_infected"]],
                is_infectable
            ),
            ChangeStateConditionalTransition(
                "will_be_hospitalized",
                ~boolean_states["will_be_hospitalized"],
                will_be_hospitalized_cond
            ),
            # TODO: This is a bit sloppy, timer shouly only run
            # for newly will be hospitalized
            InitializeTimerTransition(
                "time_until_hospitalization_init",
                timer_states["time_until_hospitalization"],
                self._infection.time_until_hospitalization,
                boolean_states["is_new_incubation"]
            ),
            MultiStateConditionalTransition(
                "will_be_hospitalized_hospitalized",
                boolean_states["will_be_hospitalized"],
                [boolean_states["is_hospitalized"],
                 boolean_states["is_new_hospitalized"],
                 boolean_states["is_removed"],
                 (~boolean_states["is_infectious"], False),
                 (~boolean_states["is_incubation"], False),
                 (~boolean_states["is_recovering"], False)],
                ~(timer_states["time_until_hospitalization"])
            ),
            InitializeTimerTransition(
                "hospitalization_duration_timer_init",
                timer_states["hospitalization_duration"],
                self._infection.hospitalization_duration,
                boolean_states["is_new_hospitalized"]
            ),

            ChangeStateConditionalTransition(
                "will_die",
                ~boolean_states["will_die"],
                will_die_cond
            ),
            # TODO: This is a bit sloppy, timer shouly only run
            # for newly will die
            InitializeTimerTransition(
                "time_until_death_init",
                timer_states["time_until_death"],
                self._infection.time_incubation_death,
                boolean_states["is_new_hospitalized"]
            ),

            MultiStateConditionalTransition(
                "will_die_is_dead",
                boolean_states["will_die"],
                [boolean_states["is_dead"],
                 boolean_states["is_removed"],
                 (~boolean_states["is_infectious"], False),
                 (~boolean_states["is_incubation"], False),
                 (~boolean_states["is_infected"], False),
                 (~boolean_states["is_hospitalized"], False),
                 (~boolean_states["is_new_hospitalized"], False)],
                ~(timer_states["time_until_death"])
            ),

            InitializeTimerTransition(
                "incubation_timer_initialization",
                timer_states["incubation_duration"],
                self._infection.incubation_duration,
                boolean_states["is_new_incubation"]
            ),
            MultiStateConditionalTransition(
                "incubation_infectious",
                boolean_states["is_incubation"],
                [boolean_states["is_infectious"],
                 boolean_states["is_new_infectious"]],
                ~(timer_states["incubation_duration"])
            ),
            InitializeTimerTransition(
                "infectious_timer_initialization",
                timer_states["infectious_duration"],
                self._infection.infectious_duration,
                boolean_states["is_new_infectious"],
            ),

            MultiStateConditionalTransition(
                "infectious_recovering",
                boolean_states["is_infectious"],
                [boolean_states["is_recovering"],
                 boolean_states["is_new_recovering"],
                 boolean_states["is_removed"]],
                normal_recovery_condition
            ),

            InitializeTimerTransition(
                "infectious_timer_initialization",
                timer_states["recovery_time"],
                self._infection.recovery_time,
                boolean_states["is_new_recovering"],
            ),

            MultiStateConditionalTransition(
                "recovering_recovered",
                boolean_states["is_recovering"],
                [boolean_states["is_recovered"],
                 (~boolean_states["is_infected"], False)],
                ~(timer_states["recovery_time"])
            ),

            MultiStateConditionalTransition(
                "hospitalized_recovered",
                boolean_states["is_hospitalized"],
                [boolean_states["is_recovered"],
                 (~boolean_states["is_infected"], False)],
                ~(timer_states["hospitalization_duration"])
            ),

            DecreaseTimerTransition(
                "decrease_incubation_time",
                timer_states["incubation_duration"],
                ~(boolean_states["is_new_incubation"])
            ),

            DecreaseTimerTransition(
                "decrease_infectious_time",
                timer_states["infectious_duration"],
                ~(boolean_states["is_new_infectious"])
            ),

            DecreaseTimerTransition(
                "decrease_recovery_time",
                timer_states["recovery_time"],
                ~(boolean_states["is_new_recovering"])
            ),

            DecreaseTimerTransition(
                "decrease_time_until_hospitalization",
                timer_states["time_until_hospitalization"],
                ~(boolean_states["is_new_incubation"])
            ),

            DecreaseTimerTransition(
                "decrease_hospitalization_duration",
                timer_states["hospitalization_duration"],
                ~(boolean_states["is_new_hospitalized"])
            ),

            DecreaseTimerTransition(
                "decrease_time_until_death",
                timer_states["time_until_death"],
                ~(boolean_states["is_new_hospitalized"])
            ),

            ChangeStateConditionalTransition(
                "deactivate_is_new_incubation",
                (boolean_states["is_new_incubation"], False),
                None
            ),
            ChangeStateConditionalTransition(
                "deactivate_is_new_infectious",
                (boolean_states["is_new_infectious"], False),
                None
            ),
            ChangeStateConditionalTransition(
                "deactivate_is_new_recovering",
                (boolean_states["is_new_recovering"], False),
                None
            ),

            ChangeStateConditionalTransition(
                "deactivate_is_new_hospizalized",
                (boolean_states["is_new_hospitalized"], False),
                None
            ),
        ]

    @property
    def transitions(self):
        return self._transitions

    @property
    def states(self):
        return self._states

    def __get_new_infections(self, data: DataDict) -> np.ndarray:
        pop_csr = self._interactions

        infected_mask = self.states["is_infected"](data)
        infected_indices = np.nonzero(infected_mask)[0]

        # Find all non-zero connections of the infected
        # rows are the ids / indices of the infected
        # columns are the people they have contact with

        _, contact_cols, contact_strengths =\
            sparse.find(pop_csr[infected_indices])

        # Based on the contact rate, sample a poisson rvs
        # for the number of interactions per timestep.
        # A contact is sucessful if the rv is > 1, ie.
        # more than one contact per timestep
        successful_contacts_mask = self._rstate.poisson(
            contact_strengths) >= 1
        # we are just interested in the columns, ie. only the
        # ids of the people contacted by the infected.
        # Note, that contacted ids can appear multiple times
        # if a person is successfully contacted by multiple people.
        successful_contacts_indices = contact_cols[successful_contacts_mask]
        num_succesful_contacts = len(successful_contacts_indices)

        self._statistics["contacts"].append(
            num_succesful_contacts)

        # Calculate infection probability for all contacts
        contact_strength = self._intensity_pdf.rvs(num_succesful_contacts)
        infection_prob = self._infection.pdf_infection_prob(contact_strength)

        # An infection is successful if the bernoulli outcome
        # based on the infection probability is 1

        newly_infected_mask = self._rstate.binomial(1, infection_prob)
        newly_infected_mask = np.asarray(newly_infected_mask, bool)

        # Get the indices for the newly infected
        newly_infected_indices = successful_contacts_indices[
            newly_infected_mask]

        # There might be multiple successfull infections per person 
        # from different infected people
        newly_infected_indices = np.unique(newly_infected_indices)
        cond = np.zeros(len(infected_mask), dtype=np.bool)
        cond[newly_infected_indices] = True

        return cond

    def __will_be_hospitalized(self, data: DataDict) -> np.ndarray:
        new_incub_indices = np.nonzero(
            self.states["is_new_incubation"](data))[0]
        if len(new_incub_indices) == 0:
            return np.zeros(data.field_len, dtype=np.bool)

        num_new_incub = len(new_incub_indices)
        will_be_hospitalized_prob = self._infection.hospitalization_prob.rvs(
                num_new_incub
            )

        # roll the dice
        will_be_hospitalized = (
            self._rstate.binomial(
                1, will_be_hospitalized_prob, size=num_new_incub
            )
            == 1
        )

        will_be_hospitalized_indices = new_incub_indices[will_be_hospitalized]
        cond = np.zeros(data.field_len, dtype=np.bool)
        cond[will_be_hospitalized_indices] = True

        return cond

    def __will_die(self, data: DataDict) -> np.ndarray:
        new_hosp_indices = np.nonzero(
            self.states["is_new_hospitalized"](data))[0]
        if len(new_hosp_indices) == 0:
            return np.zeros(data.field_len, dtype=np.bool)

        num_new_incub = len(new_hosp_indices)
        will_die_prob = self._infection.death_prob.rvs(
                num_new_incub
            )

        # roll the dice
        will_die = (
            self._rstate.binomial(
                1, will_die_prob, size=num_new_incub
            )
            == 1
        )

        will_die_indices = new_hosp_indices[will_die]
        cond = np.zeros(data.field_len, dtype=np.bool)
        cond[will_die_indices] = True

        return cond
