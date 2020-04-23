# -*- coding: utf-8 -*-

"""
Name: state_machine.py
Authors: Christian Haack, Stephan Meighen-Berger, Andrea Turcati
Constructs the state machine
"""

from __future__ import annotations
import abc
from collections import defaultdict
import functools
import logging
from typing import Callable, Union, List, Tuple, Dict, Optional

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from .config import config
from .infection import Infection
from .measures import Measures
from .pdfs import PDF
from .population import Population


_log = logging.getLogger(__name__)

DEBUG = False

if DEBUG:
    _log.warn("DEBUG flag enabled. This will drastically slow down the code")


class DataDict(dict):
    """
    Dictionary of numpy arrays with equal length

    Attributes:
        field_len: int
            Length of the arrays stored in the `DataDict`
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        field_len = -1
        for key, val in self.items():
            if field_len >= 0 and len(val) != field_len:
                raise RuntimeError("Not all fields are of same length")
            field_len = len(val)

        self._field_len = field_len

    @property
    def field_len(self) -> int:
        return self._field_len


class Condition(object):
    """
    Convenience class for storing conditions

    Parameters:
        condition: Callable[[DataDict], np.ndarray]
            A callable that calculates the condition on a `DataDict`
            and returns a boolean numpy array encoding on which row
            the condition is met.

    Attributes:
        condition

    """

    def __init__(self, condition: Callable[[DataDict], np.ndarray]) -> None:
        self._condition = condition

    @classmethod
    def from_state(cls, state: _State):
        """
        Factory method for instantiating a `Condition` from a `_State`

        Parameters:
            state: _State
        """

        return cls(state)

    @property
    def condition(self) -> Callable[[DataDict], np.ndarray]:
        return self._condition

    @condition.setter
    def condition(self, val: Callable[[DataDict], np.ndarray]) -> None:
        self._condition = val

    def __call__(self, data: DataDict):
        """
        Evaluate condition on a `DataDict`

        Parameters:
            data: DataDict
        """
        return self.condition(data)

    def __and__(self, other: TCondition) -> Condition:
        """
        Logical and of a condition and an object of type `TCondition`

        Parameters:
            other: TCondition

        """

        def new_condition(data: DataDict):
            cond = unify_condition(other, data)

            return self(data) & cond

        return Condition(new_condition)


TCondition = Union["_State", np.ndarray, Condition]


def unify_condition(condition: TCondition, data: DataDict) -> np.ndarray:
    """
    Convenience function to convert a condition of type `TCondition` to array

    Parameters:
        condition: TCondition
        data: DataDict

    Returns:
        np.ndarray
    """
    if isinstance(condition, (_State, Condition)):
        # Evaluate on data to get condition array
        cond = condition(data)
    elif isinstance(condition, np.ndarray):
        cond = condition
    elif condition is None:
        cond = np.ones(data.field_len, dtype=np.bool)
    else:
        raise ValueError("Unsupported type: ", type(condition))
    return cond


class ConditionalMixin(object):
    """Mixin class for storing conditions"""

    def __init__(self, condition: TCondition, *args, **kwargs):
        self._condition = condition

    def unify_condition(self, data: DataDict):
        """
        Wrapper for `unify_condition`

        Calls `unify_condition` with the stored condition
        """
        return unify_condition(self._condition, data)


class _State(object, metaclass=abc.ABCMeta):
    """
    Metaclass for States

    Parameters:
        state_getter: Callable[[np.ndarray], np.ndarray]
            A callable that takes an array as single argument and returns
            an boolean array that encodes which rows are in this state
        state_getter: Callable[[np.ndarray], np.ndarray]
            A callable that takes an array as single argument and returns
            an array that encodes the state value of each row
        name: str
            The state name
        data_field: str
            The data field name which stores this state
        state_change:  Callable[[DataDict, np.ndarray, TCondition], None]
            A callable that changes the state in the `DataDict` based on a
            TCondition. The callable takes a DataDict as first argument,
            a numpy.ndarray storing the target states as second, and a
            TCondition as third argument

    """

    def __init__(
        self,
        state_getter: Callable,
        state_value_getter: Callable,
        name: str,
        data_field: str,
        state_change: Callable,
        *args,
        **kwargs,
    ):

        self._state_getter = state_getter
        self._state_value_getter = state_value_getter
        self._name = name
        self._state_change = state_change
        self._data_field = data_field

    def __call__(self, data: DataDict) -> np.ndarray:
        """
        Returns the state

        Parameters:
            data: DataDict

        Returns:
            np.ndarray
        """
        return self._state_getter(data[self._data_field])

    def get_state_value(self, data: DataDict) -> np.ndarray:
        """Returns the state values"""
        return self._state_value_getter(data[self._data_field])

    def __invert__(self) -> _State:
        """
        Return an inverted state

        Calls the state_getter of the original state and inverts
        the resulting numpy array

        """

        def inverted_condition(arr: np.ndarray):
            return ~(self._state_getter(arr))

        return type(self)(
            inverted_condition,
            self._state_value_getter,
            "inverted_" + self.name,
            self._data_field,
            self._state_change,
        )

    def change_state(
        self, data: DataDict, state: np.ndarray, condition: TCondition = None,
    ) -> None:
        """
        Changes the state in the DataDict

        Parameters:
            data: DataDict
            state: np.ndarray
                The target state values
            condition: Optional[TCondition]
        """

        self_cond = self(data)
        cond = unify_condition(condition, data)
        self._state_change(data, state, cond & self_cond)

    @property
    def name(self):
        return self._name


class BooleanState(_State):
    """
    Specialization for boolean states.
    """

    @classmethod
    def from_boolean(cls, name: str) -> BooleanState:
        """
        Factory method for creating a state from a boolean field
        in a DataDict. The name of the state corresponds to the data field name
        in the DataDict.

        Parameters:
            name: str
        """

        def get_state(arr: np.ndarray):
            return arr

        def state_change(
            data: DataDict, state: np.ndarray, condition: np.ndarray
        ):

            # TODO: maybe offload application of condition to state here?
            data[name][condition] = state

        return cls(get_state, get_state, name, name, state_change)


class FloatState(_State):
    """
    Specialization for a float state
    """

    @classmethod
    def from_timer(cls, name: str) -> FloatState:
        """
        Factory method for creating a state from a timer field
        in a DataDict.  The name of the state corresponds to the data field
        name in the DataDict.

        Parameters:
            name: str
        """

        def get_state(arr: np.ndarray):
            # State is active when field is > 0
            return arr > 0

        def get_state_value(arr: np.ndarray):
            return arr

        def state_change(data: DataDict, state: np.ndarray, condition):
            data[name][condition] = state

        return cls(get_state, get_state_value, name, name, state_change)

    @classmethod
    def from_counter(cls, name: str) -> FloatState:
        """
        Factory method for creating a state from a counter field
        in a DataDict.  The name of the state corresponds to the data field
        name in the DataDict.

        Parameters:
            name: str
        """

        def get_state(arr: np.ndarray):
            # State is active when field is > -np.inf
            return arr > -np.inf

        def get_state_value(arr: np.ndarray):
            return arr

        def state_change(data: DataDict, state: np.ndarray, condition):
            data[name][condition] = state

        return cls(get_state, get_state_value, name, name, state_change)


def log_call(func):
    """
    Convenience function for logging

    When `DEBUG` is set to true, log the difference of the
    DataDict after each transition
    """
    if DEBUG:

        @functools.wraps(func)
        def log_wrapper(self, data):
            _log.debug("Performing %s", self.name)
            df_before = pd.DataFrame(data, copy=True)
            retval = func(self, data)
            df_after = pd.DataFrame(data, copy=True)

            diff = df_before.astype("float") - df_after.astype("float")

            diff_rows = diff.loc[diff.any(axis=1), :]
            diff_cols = diff_rows.loc[:, diff_rows.any(axis=0)]

            df_after = df_after.loc[diff.any(axis=1), :]
            df_after = df_after.loc[:, diff_rows.any(axis=0)]
            _log.debug("Dataframe diff: %s", diff_cols)
            _log.debug("Dataframe now: %s", df_after)

            return retval

        return log_wrapper
    else:
        return func


class _Transition(object, metaclass=abc.ABCMeta):
    """
    Metaclass for Transitions.

    Subclasses have to implement a `__call__` method that performs the
    transition.

    Parameters:
        name: str
    """

    def __init__(self, name: str, *args, **kwargs):
        self._name = name

    @abc.abstractmethod
    def __call__(self, data: DataDict) -> None:
        pass

    @property
    def name(self) -> str:
        return self._name


class Transition(_Transition):
    def __init__(
        self, name: str, state_a: _State, state_b: _State, *args, **kwargs
    ):

        super().__init__(name, *args, **kwargs)
        self._state_a = state_a
        self._state_b = state_b

    @log_call
    def __call__(self, data: DataDict):
        """
        Perform the transition.

        All rows in data which where previusly in state A are transitioned
        to state B.

        Parameters:
            data: DataDict
        """

        # Invert state B to select all rows which are _not_ in state B
        # Use state A as condition so that only rows are activated which
        # where in state A

        (~self._state_b).change_state(data, True, self._state_a(data))
        self._state_a.change_state(data, False)


class ChangeStatesConditionalTransition(_Transition, ConditionalMixin):
    """
    Change a state where external condition is true

    Parameters:
        name: str
        state_a: Union[_State, Tuple[_State, bool]]
            State to change. Can either be a `_State`, in which case the state
            will be set to true or a Tuple[_State, bool], where the second item
            is the value the state should be set to.

    """

    _states_a: List[_State]
    _states_a_vals: List[bool]

    def __init__(
        self,
        name: str,
        states_a: Union[
            Union[_State, Tuple[_State, bool]],
            List[Union[_State, Tuple[_State, bool]]],
        ],
        condition: TCondition,
        *args,
        **kwargs,
    ):
        _Transition.__init__(self, name, *args, **kwargs)
        ConditionalMixin.__init__(self, condition, *args, **kwargs)

        if not isinstance(states_a, list):
            states_a = [states_a]

        self._states_a = []
        self._states_a_vals = []

        for state in states_a:
            if isinstance(state, tuple):
                self._states_a.append(state[0])
                self._states_a_vals.append(state[1])
            else:
                self._states_a.append(state)
                self._states_a_vals.append(True)

    @log_call
    def __call__(self, data: DataDict):
        cond = self.unify_condition(data)

        for state, val in zip(self._states_a, self._states_a_vals):
            state.change_state(data, val, cond)

        return cond


class ConditionalTransition(_Transition, ConditionalMixin):
    """
    Perform a transition from state_a to state_b where an external condition
    is true

    Parameters:
        name: str
        state_a: _State
        state_b: _State
        condition: TCondition
    """

    def __init__(
        self,
        name: str,
        state_a: _State,
        state_b: _State,
        condition: TCondition,
        *args,
        **kwargs,
    ):

        _Transition.__init__(self, name, *args, **kwargs)
        ConditionalMixin.__init__(self, condition, *args, **kwargs)
        self._state_a = state_a
        self._state_b = state_b

    @log_call
    def __call__(self, data: DataDict):

        cond = self.unify_condition(data)

        (~self._state_b).change_state(data, True, cond & self._state_a(data))
        self._state_a.change_state(data, False, cond)

        return cond


class DecreaseTimerTransition(_Transition, ConditionalMixin):
    """
    Decrease the value of a FloatState by one

    Parameters:
        name: str
        state_a: FloatState
        condition: Optional[TCondition]
    """

    def __init__(
        self,
        name: str,
        state_a: FloatState,
        condition: TCondition = None,
        *args,
        **kwargs,
    ):
        _Transition.__init__(self, name, *args, **kwargs)
        ConditionalMixin.__init__(self, condition, *args, **kwargs)
        self._state_a = state_a

    @log_call
    def __call__(self, data: DataDict):

        cond = unify_condition(self._condition, data)
        state_condition = self._state_a(data)

        # Current state value
        cur_state = self._state_a.get_state_value(data)[cond & state_condition]
        self._state_a.change_state(data, cur_state - 1, cond)


class IncreaseTimerTransition(_Transition, ConditionalMixin):
    """
    Increase the value of a FloatState by one

    Parameters:
        name: str
        state_a: FloatState
        condition: Optional[TCondition]
    """

    def __init__(
        self,
        name: str,
        state_a: FloatState,
        condition: TCondition = None,
        *args,
        **kwargs,
    ):
        _Transition.__init__(self, name, *args, **kwargs)
        ConditionalMixin.__init__(self, condition, *args, **kwargs)
        self._state_a = state_a

    @log_call
    def __call__(self, data: DataDict):
        cond = unify_condition(self._condition, data)
        state_condition = self._state_a(data)

        # Current state value
        cur_state = self._state_a.get_state_value(data)[cond & state_condition]
        self._state_a.change_state(data, cur_state + 1, cond)


class InitializeTimerTransition(_Transition, ConditionalMixin):
    """
    Initialize a timer state to values drawn from a PDF

    Parameters:
        name: str
        state_a: FloatState
        initialization_pdf: PDF
        condition: Optional[TCondition]
    """

    def __init__(
        self,
        name: str,
        state_a: FloatState,
        initialization_pdf: PDF,
        condition: TCondition = None,
        *args,
        **kwargs,
    ):

        _Transition.__init__(self, name, *args, **kwargs)
        ConditionalMixin.__init__(self, condition)
        self._state_a = state_a
        self._initialization_pdf = initialization_pdf

    @log_call
    def __call__(self, data: DataDict):

        cond = self.unify_condition(data)

        # Rows which are currently 0
        zero_rows = (~self._state_a(data)) & cond
        num_zero_rows = zero_rows.sum(axis=0)

        initial_vals = self._initialization_pdf.rvs(num_zero_rows)
        (~self._state_a).change_state(data, initial_vals, cond)


class InitializeCounterTransition(_Transition, ConditionalMixin):
    """
    Initialize a counter state

    Parameters:
        name: str
        state_a: FloatState
        start: int
        condition: Optional[TCondition]
    """

    def __init__(
        self,
        name: str,
        state_a: FloatState,
        start: int = 0,
        condition: TCondition = None,
        *args,
        **kwargs,
    ):

        _Transition.__init__(self, name, *args, **kwargs)
        ConditionalMixin.__init__(self, condition)
        self._state_a = state_a
        self._start = start

    @log_call
    def __call__(self, data: DataDict):

        cond = self.unify_condition(data)

        initial_vals = self._start  # Initializes counter at 1
        (~self._state_a).change_state(data, initial_vals, cond)


class MultiStateConditionalTransition(_Transition, ConditionalMixin):
    """
    Perform a transition from state_a to multiple other states

    Parameters:
        name: str
        state_a:  Union[_State, Tuple[_State, bool]]
            State to change. Can either be a `_State`, in which case the state
            will be set to true or a Tuple[_State, bool], where the second item
            is the value the state should be set to.
        states_b: List[Union[_State, Tuple[_State, bool]]]
            List of states to change. Can either be a `_State`, in which case
                the state will be set to true or a Tuple[_State, bool], where
                the second item is the value the state should be set to.
    """

    _states_b: List[_State]
    _states_b_vals: List[bool]

    def __init__(
        self,
        name: str,
        state_a: Union[_State, Tuple[_State, bool]],
        states_b: List[Union[_State, Tuple[_State, bool]]],
        condition: TCondition,
        *args,
        **kwargs,
    ):

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
            (~state).change_state(data, val, cond & is_in_state_a)
        self._state_a.change_state(data, self._state_a_val, cond)

        return cond


class StatCollector(object, metaclass=abc.ABCMeta):
    """
    Convenience class for collecting statistics

    Parameters:
        data_fields: List[str]
            List of data field names to track
    """

    _statistics: Dict[str, List[float]]

    def __init__(self, data_fields: List[str]):
        self._data_fields = data_fields
        self._statistics = defaultdict(list)

    def __call__(self, data: DataDict):
        for field in self._data_fields:
            self._statistics[field].append(data[field].sum())

    @property
    def statistics(self):
        return self._statistics


class StateMachine(object, metaclass=abc.ABCMeta):
    """
    Base class for state machines

    Subclasses have to implement the `transitions` and `state` properties.

    Parameters:
        data: Union[pd.DataFrame, DataDict]
            Can be either a DataFrame of `DataDict`
        stat_collection: Optional[StatCollector]
            Stat collector object
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, DataDict],
        stat_collector: Optional[StatCollector],
        *args,
        **kwargs,
    ):
        if isinstance(data, pd.DataFrame):
            self._data = DataDict(
                {key: data[key].values for key in data.columns}
            )
        else:
            self._data = data
        self._stat_collector = stat_collector
        if config["general"]["trace spread"]:
            _log.debug("Tracing the population")
            self._trace_contacts = []
            self._trace_infection = []

    @property
    @abc.abstractmethod
    def transitions(self) -> List[_Transition]:
        pass

    @property
    @abc.abstractmethod
    def states(self) -> List[_State]:
        pass

    def tick(self) -> None:
        """
        Perform all transitions
        """
        for transition in self.transitions:
            transition(self._data)
        if self._stat_collector is not None:
            self._stat_collector(self._data)

    @property
    def statistics(self):
        return self._stat_collector.statistics


class ContagionStateMachine(StateMachine):
    """
    Contagion state machine

    Parameters:
        data: Union[pd.DataFrame, DataDict]
        stat_collector: Optional[StatCollector]
        interactions: sparse.spmatrix
        infection: Infection
        intensity_pdf: PDF,
        rstate: np.random.RandomState
    """

    _states: Dict[str, _State]
    _statistics: Dict[str, List[float]]

    def __init__(
        self,
        data: Union[pd.DataFrame, DataDict],
        stat_collector: Optional[StatCollector],
        population: Population,
        infection: Infection,
        measures: Measures,
        *args,
        **kwargs,
    ):
        super().__init__(data, stat_collector, *args, **kwargs)

        self._population = population
        self._rstate = config["runtime"]["random state"]
        self._infection = infection
        self._statistics = defaultdict(list)
        self._measures = measures

        # Boolean states
        boolean_state_names = [
            "is_infected",
            "is_new_infected",
            "is_latent",
            "is_new_latent",
            "will_have_symptoms",
            "will_have_symptoms_new",
            "is_symptomatic",
            "is_infectious",
            "is_new_infectious",
            "is_hospitalized",
            "is_new_hospitalized",
            "will_be_hospitalized",
            "will_be_hospitalized_new",
            "is_dead",
            "is_removed",
            "is_quarantined",
            "is_new_quarantined",
            "is_tracked",
            "is_recovering",
            "is_new_recovering",
            "is_recovered",
            "will_die",
            "will_die_new",
            "will_be_tested",
            "will_be_tested_new",
            "is_tested",
            "is_new_tested",
            "will_test_negative",
            "will_test_negative_new",
        ]

        boolean_states = {
            name: BooleanState.from_boolean(name)
            for name in boolean_state_names
        }

        # Timer states
        timer_state_names = [
            "latent_duration",
            "time_until_symptoms",
            "infectious_duration",
            "hospitalization_duration",
            "recovery_time",
            "time_until_hospitalization",
            "time_until_death",
            "quarantine_duration",
            "time_until_test",
            "time_until_test_result",
        ]

        timer_states = {
            name: FloatState.from_timer(name) for name in timer_state_names
        }

        # Counter states
        counter_state_names = [
            "time_since_infectious",
            "time_since_quarantine",
        ]

        counter_states = {
            name: FloatState.from_counter(name) for name in counter_state_names
        }

        self._states = {}
        self._states.update(boolean_states)
        self._states.update(timer_states)
        self._states.update(counter_states)

        # Condition that stores the new infections from this tick
        infected_condition = Condition(self.__get_new_infections)

        # Symptomatic
        will_have_symptoms_cond = Condition(self.__will_have_symptoms)

        # Condition that stores people who will be hostpialized
        will_be_hospitalized_cond = Condition(self.__will_be_hospitalized)

        # Condition that stores people who will die
        will_die_cond = Condition(self.__will_die)

        hospit_cond = Condition.from_state(
            ~timer_states["time_until_hospitalization"]
        ) & Condition.from_state(~boolean_states["will_be_hospitalized_new"])

        infectious_cond = Condition.from_state(
            ~timer_states["latent_duration"]
        ) & Condition.from_state(~boolean_states["is_new_latent"])

        symptomatic_cond = Condition.from_state(
            ~timer_states["time_until_symptoms"]
        ) & Condition.from_state(~boolean_states["will_have_symptoms_new"])

        recovery_cond = Condition.from_state(
            ~timer_states["recovery_time"]
        ) & Condition.from_state(~boolean_states["is_new_recovering"])

        is_dead_cond = Condition.from_state(
            ~timer_states["time_until_death"]
        ) & Condition.from_state(~boolean_states["will_die_new"])

        # Only people who are not hospitalized undergo normal recovery

        normal_recovery_condition = (
            Condition.from_state(~(boolean_states["is_hospitalized"]))
            & Condition.from_state(~timer_states["infectious_duration"])
            & Condition.from_state(~boolean_states["is_new_infectious"])
        )

        hospit_recovery_condition = Condition.from_state(
            ~timer_states["hospitalization_duration"]
        ) & Condition.from_state(~boolean_states["is_new_hospitalized"])

        # Quarantine condition
        quarantine_condition = Condition(
            self.__will_be_quarantined
        ) & Condition.from_state(~(boolean_states["is_removed"]))

        will_be_tested_cond = Condition(self.__will_be_tested)

        is_tested_cond = Condition.from_state(
            ~timer_states["time_until_test"]
        ) & Condition.from_state(~boolean_states["will_be_tested_new"])

        # Test negative conditions
        will_test_negative_cond = Condition(self.__will_test_negative)

        tested_negative_cond = Condition.from_state(
            ~timer_states["time_until_test_result"]
        ) & Condition.from_state(~boolean_states["will_test_negative_new"])

        quarantine_recovered_condition = Condition.from_state(
            ~timer_states["quarantine_duration"]
        ) & Condition.from_state(~boolean_states["is_new_quarantined"])

        temp_states = [
            "is_new_latent",
            "is_new_infectious",
            "will_have_symptoms_new",
            "is_new_recovering",
            "will_be_hospitalized_new",
            "is_new_hospitalized",
            "will_die_new",
            "is_new_quarantined",
            "will_be_tested_new",
            "is_new_tested",
            "will_test_negative_new",
        ]

        # Timer name, tick when not in this state
        # state will be inverted
        timer_ticks = [
            (
                "latent_duration",
                "is_new_latent",
                self._infection.latent_duration,
            ),
            (
                "infectious_duration",
                "is_new_infectious",
                self._infection.infectious_duration,
            ),
            (
                "time_until_symptoms",
                "will_have_symptoms_new",
                self._infection.incubation_duration,
            ),
            (
                "recovery_time",
                "is_new_recovering",
                self._infection.recovery_time,
            ),
            (
                "time_until_hospitalization",
                "will_be_hospitalized_new",
                self._infection.time_until_hospitalization,
            ),
            (
                "hospitalization_duration",
                "is_new_hospitalized",
                self._infection.hospitalization_duration,
            ),
            (
                "time_until_death",
                "will_die_new",
                self._infection.time_incubation_death,
            ),
            (
                "quarantine_duration",
                "is_new_quarantined",
                self._measures.quarantine_duration,
            ),
            (
                "time_until_test",
                "will_be_tested_new",
                self._measures.time_until_test,
            ),
            (
                "time_until_test_result",
                "will_test_negative_new",
                self._measures.time_until_test_result,
            ),
        ]

        # Counter name, tick when not in this state
        counter_ticks = [
            ("time_since_infectious", "is_new_infectious"),
            ("time_since_quarantine", "is_new_quarantined"),
        ]

        # Transitions
        self._transitions = [
            # Healthy - latent
            # Transition from not-infected to:
            #   - is_latent
            #   - is_new_latent
            #   - is_infected
            # if the is infectable condition is true
            MultiStateConditionalTransition(
                "healthy_latent",
                ~boolean_states["is_infected"],
                [
                    boolean_states["is_latent"],
                    boolean_states["is_new_latent"],
                    boolean_states["is_infected"],
                ],
                infected_condition,
            ),
            # When latent time is over, people become infectious
            MultiStateConditionalTransition(
                "latent_infectious",
                boolean_states["is_latent"],
                [
                    boolean_states["is_infectious"],
                    boolean_states["is_new_infectious"],
                ],
                infectious_cond,
            ),
            # Determine if ppl will develop symptoms
            ChangeStatesConditionalTransition(
                "will_have_symptoms",
                [
                    ~boolean_states["will_have_symptoms"],
                    ~boolean_states["will_have_symptoms_new"],
                ],
                will_have_symptoms_cond,
            ),
            # No symptoms - symptomatic
            ConditionalTransition(
                "no_symptoms_symptomatic",
                boolean_states["will_have_symptoms"],
                boolean_states["is_symptomatic"],
                symptomatic_cond,
            ),
            # infectious - recovering
            # Transition from is_infectious to:
            #   -is_recovering
            #   -is_new_recovering
            #   -is_removed
            # where the normal_recovery_condition is True
            MultiStateConditionalTransition(
                "infectious_recovering",
                boolean_states["is_infectious"],
                [
                    boolean_states["is_recovering"],
                    boolean_states["is_new_recovering"],
                    boolean_states["is_removed"],
                    (~boolean_states["is_symptomatic"], False),
                    (~boolean_states["will_have_symptoms"], False),
                    (~counter_states["time_since_infectious"], -np.inf),
                ],
                normal_recovery_condition,
            ),
            # recovering - recovered
            # Transition from is_recovering to:
            #   -is_recovered
            #   -not is_infected
            # where the recovery_time timer is <= 0
            MultiStateConditionalTransition(
                "recovering_recovered",
                boolean_states["is_recovering"],
                [
                    boolean_states["is_recovered"],
                    (~boolean_states["is_infected"], False),
                ],
                recovery_cond,
            ),
            # Activate will_be_hospitalized and will_be_hospitalized_new if the
            # will_be_hospitalized_cond condition is true.
            ChangeStatesConditionalTransition(
                "will_be_hospitalized",
                [
                    ~boolean_states["will_be_hospitalized"],
                    ~boolean_states["will_be_hospitalized_new"],
                ],
                will_be_hospitalized_cond,
            ),
            # will_be_hospitalized - hospitalized
            # Transition from will_be_hospitalized to:
            #   -is_hospitalized
            #   -is_new_hospitalized
            #   -is_removed
            #   -not is_recovering
            # where the time_until_hospitalization timer is <= 0
            MultiStateConditionalTransition(
                "will_be_hospitalized_hospitalized",
                boolean_states["will_be_hospitalized"],
                [
                    boolean_states["is_hospitalized"],
                    boolean_states["is_new_hospitalized"],
                    boolean_states["is_removed"],
                    (~boolean_states["is_quarantined"], False),
                ],
                hospit_cond,
            ),
            MultiStateConditionalTransition(
                "hospitalized_recovered",
                boolean_states["is_hospitalized"],
                [
                    boolean_states["is_recovered"],
                    (~boolean_states["is_infected"], False),
                    (~boolean_states["is_symptomatic"], False),
                    (~boolean_states["will_have_symptoms"], False),
                ],
                hospit_recovery_condition,
            ),
            # Activate will_die and will_die_new if the
            # will_die_cond condition is true.
            ChangeStatesConditionalTransition(
                "will_die",
                [~boolean_states["will_die"], ~boolean_states["will_die_new"]],
                will_die_cond,
            ),
            # will_die - is_dead
            # Transition from will_die to:
            #   -is_dead
            #   -is_removed
            #   -not is_infected
            #   -not is_hospitalized
            #   -not is_new_hospitalized
            # where the time_until_death timer is <= 0
            MultiStateConditionalTransition(
                "will_die_is_dead",
                boolean_states["will_die"],
                [
                    boolean_states["is_dead"],
                    boolean_states["is_removed"],
                    (~boolean_states["is_infectious"], False),
                    (~boolean_states["is_infected"], False),
                    (~boolean_states["is_hospitalized"], False),
                    (~boolean_states["is_new_hospitalized"], False),
                    (~boolean_states["is_symptomatic"], False),
                    (~boolean_states["will_have_symptoms"], False),
                ],
                is_dead_cond,
            ),
            # Susceptible - Quarantine
            # transition if:
            #  - is_infected
            #  - is_tracked
            MultiStateConditionalTransition(
                "is_quarantined",
                ~boolean_states["is_quarantined"],
                [
                    boolean_states["is_quarantined"],
                    boolean_states["is_new_quarantined"],
                    boolean_states["is_removed"],
                ],
                quarantine_condition,
            ),
            ChangeStatesConditionalTransition(
                "will_be_tested",
                [
                    ~boolean_states["will_be_tested"],
                    ~boolean_states["will_be_tested_new"],
                ],
                will_be_tested_cond,
            ),
            # will_be_tested -> tested
            MultiStateConditionalTransition(
                "will_be_tested_tested",
                boolean_states["will_be_tested"],
                [
                    boolean_states["is_tested"],
                    boolean_states["is_new_tested"],
                ],
                is_tested_cond,
            ),
            ChangeStatesConditionalTransition(
                "will_test_negative",
                [
                    ~boolean_states["will_test_negative"],
                    ~boolean_states["will_test_negative_new"],
                ],
                will_test_negative_cond,
            ),
            MultiStateConditionalTransition(
                "tested_negative",
                boolean_states["will_test_negative"],
                [
                    (~boolean_states["is_tested"], False),
                    (~boolean_states["is_removed"], False),
                    (~boolean_states["is_quarantined"], False),
                    (~counter_states["time_since_quarantine"], -np.inf),
                ],
                tested_negative_cond,
            ),
            # TODO: Does coming out of qurantine mean that you are healed?
            # Go out of quarantine
            MultiStateConditionalTransition(
                "quarantine_recovered",
                boolean_states["is_quarantined"],
                [
                    (~boolean_states["is_infected"], False),
                    boolean_states["is_recovered"],
                    (~boolean_states["is_symptomatic"], False),
                    (~boolean_states["is_infectious"], False),
                    (~boolean_states["will_be_hospitalized"], False),
                    (~boolean_states["will_have_symptoms"], False),
                    (~counter_states["time_since_quarantine"], -np.inf),
                ],
                quarantine_recovered_condition,
            ),
        ]

        for (state, dont_tick_when, init_func) in timer_ticks:
            self._transitions.append(
                InitializeTimerTransition(
                    "init_{}".format(state),
                    timer_states[state],
                    init_func,
                    boolean_states[dont_tick_when],
                )
            )
            self._transitions.append(
                DecreaseTimerTransition(
                    "decrease_{}".format(state),
                    timer_states[state],
                    ~(boolean_states[dont_tick_when]),
                )
            )

        for (state, dont_tick_when) in counter_ticks:
            self._transitions.append(
                InitializeCounterTransition(
                    "init_{}".format(state),
                    counter_states[state],
                    1,
                    boolean_states[dont_tick_when],
                )
            )
            self._transitions.append(
                IncreaseTimerTransition(
                    "increase_{}".format(state),
                    counter_states[state],
                    ~(boolean_states[dont_tick_when]),
                )
            )

        # Deactivate temp states
        for state in temp_states:
            self._transitions.append(
                ChangeStatesConditionalTransition(
                    "deactivate_{}".format(state),
                    (boolean_states[state], False),
                    None,
                )
            )

    @property
    def trace_contacts(self):
        return self._trace_contacts

    @property
    def trace_infection(self):
        return self._trace_infection

    @property
    def transitions(self):
        return self._transitions

    @property
    def states(self):
        return self._states

    def __get_new_infections(self, data: DataDict) -> np.ndarray:

        infectious_mask = self.states["is_infectious"](data) & ~self.states[
            "is_new_infectious"
        ](data)

        if infectious_mask.sum() == 0:
            return np.zeros_like(infectious_mask, dtype=np.bool)

        # Only infectious non removed can infect others
        removed_mask = self.states["is_removed"](data)
        infectious_mask = infectious_mask & (~removed_mask)
        infectious_indices = np.nonzero(infectious_mask)[0]

        healthy_mask = ~self.states["is_infected"](data)
        is_infectable = healthy_mask & (~removed_mask)
        is_infectable_indices = np.nonzero(is_infectable)[0]

        if len(is_infectable_indices) == 0:
            return np.zeros_like(infectious_mask, dtype=np.bool)

        # NOTE: This is ~2times slower
        (
            contact_cols,
            contact_strengths,
            contact_rows,
        ) = self._population.get_contacts(
            infectious_indices, is_infectable_indices, return_rows=True
        )
        # contacted_mask = np.zeros_like(infectious_mask)
        # contacted_mask[contact_cols] = True

        # Based on the contact rate, sample a poisson rvs
        # for the number of interactions per timestep.
        # A contact is sucessful if the rv is > 1, ie.
        # more than one contact per timestep
        successful_contacts_mask = self._rstate.poisson(contact_strengths) >= 1

        # check if the successful contacts are quarantined

        # we are just interested in the columns, ie. only the
        # ids of the people contacted by the infected.
        # Note, that contacted ids can appear multiple times
        # if a person is successfully contacted by multiple people.
        successful_contacts_indices = contact_cols[successful_contacts_mask]
        successful_contactee_indices = contact_rows[successful_contacts_mask]

        # Calculate infection probability for all contacts
        # The duration of each infection
        infectious_dur = data["time_since_infectious"][
            successful_contactee_indices
        ]

        infection_prob = self._infection.pdf_infection_prob.pdf(infectious_dur)

        # An infection is successful if the bernoulli outcome
        # based on the infection probability is 1

        newly_infected_mask = self._rstate.binomial(1, infection_prob)
        newly_infected_mask = np.asarray(newly_infected_mask, bool)
        # Get the indices for the newly infected
        newly_infected_indices = successful_contacts_indices[
            newly_infected_mask
        ]

        # TODO: Add this as a state not seperate list
        if config["general"]["trace spread"]:
            self._trace_contacts.append(
                np.dstack(
                    (
                        successful_contactee_indices,
                        successful_contacts_indices,
                    )
                )
            )
        num_succesful_contacts = len(successful_contacts_indices)
        self._statistics["contacts"].append(num_succesful_contacts)

        if config["general"]["trace spread"]:

            newly_infectee_indices = successful_contactee_indices[
                newly_infected_mask
            ]

            self._trace_infection.append(
                np.dstack((newly_infectee_indices, newly_infected_indices))
            )

        # There might be multiple successfull infections per person
        # from different infected people
        newly_infected_indices = np.unique(newly_infected_indices)
        cond = np.zeros_like(infectious_mask, dtype=np.bool)
        cond[newly_infected_indices] = True

        return cond

    def __will_be_hospitalized(self, data: DataDict) -> np.ndarray:
        new_indices = np.nonzero(self.states["will_have_symptoms_new"](data))[
            0
        ]
        if len(new_indices) == 0:
            return np.zeros(data.field_len, dtype=np.bool)

        num_new = len(new_indices)
        will_be_hospitalized_prob = self._infection.hospitalization_prob.rvs(
            num_new
        )

        # roll the dice
        will_be_hospitalized = (
            self._rstate.binomial(1, will_be_hospitalized_prob, size=num_new)
            == 1
        )

        will_be_hospitalized_indices = new_indices[will_be_hospitalized]

        cond = np.zeros(data.field_len, dtype=np.bool)
        cond[will_be_hospitalized_indices] = True

        return cond

    def __will_die(self, data: DataDict) -> np.ndarray:
        new_hosp_indices = np.nonzero(
            self.states["is_new_hospitalized"](data)
        )[0]
        if len(new_hosp_indices) == 0:
            return np.zeros(data.field_len, dtype=np.bool)

        num_new_incub = len(new_hosp_indices)
        will_die_prob = self._infection.death_prob.rvs(num_new_incub)

        # roll the dice
        will_die = (
            self._rstate.binomial(1, will_die_prob, size=num_new_incub) == 1
        )

        will_die_indices = new_hosp_indices[will_die]
        cond = np.zeros(data.field_len, dtype=np.bool)
        cond[will_die_indices] = True

        return cond

    def __will_have_symptoms(self, data: DataDict) -> np.ndarray:

        new_infec = data["is_new_infectious"]
        num_new_infec = new_infec.sum()
        new_infec_indices = new_infec.nonzero()[0]
        symp_prob = self._infection.will_have_symptoms_prob.rvs(num_new_infec)

        will_have_symp = (
            self._rstate.binomial(1, symp_prob, size=num_new_infec) == 1
        )

        will_have_symp_indices = new_infec_indices[will_have_symp]
        cond = np.zeros(data.field_len, dtype=np.bool)
        cond[will_have_symp_indices] = True

        return cond

    def __will_be_quarantined(self, data: DataDict) -> np.ndarray:
        # If you are tracked and infected you will be quarantined
        infected_mask = self.states["is_symptomatic"](data)

        if self._measures.tracked is not None:
            tracked_mask = self.states["is_tracked"](data)
        else:
            tracked_mask = np.zeros_like(infected_mask, dtype=np.bool)

        tracked_infected_mask = infected_mask & tracked_mask
        if not np.any(tracked_infected_mask):
            return tracked_infected_mask

        tracked_infected_indices = np.nonzero(tracked_infected_mask)[0]

        backtrack_length = self._measures.backtrack_length
        if backtrack_length == 0:
            return tracked_infected_mask

        # Get contact history
        con_history = self.trace_contacts
        len_history = len(con_history)

        if len_history < backtrack_length:
            backtrack_length = len_history

        if backtrack_length == 0:
            return tracked_infected_mask

        con_history = np.squeeze(np.hstack(con_history[-backtrack_length:]))

        if con_history.size == 0:
            return tracked_infected_mask

        # Only check contacts of tracked infected people
        if len(con_history) > 2:
            # Get history of tracked
            contacted_indices = []
            for tracked_id in tracked_infected_indices:
                hist_id = np.nonzero(con_history[:, 0])[0]
                contacted_indices.append(con_history[hist_id, 1])

            contacted_indices = np.unique(np.concatenate(contacted_indices))
        else:
            contacted_indices = np.unique(con_history)

        if contacted_indices.size == 0:
            return tracked_infected_mask

        contacted_mask = np.zeros_like(infected_mask, dtype=np.bool)
        contacted_mask[contacted_indices] = True
        contacted_mask[~tracked_mask] = False

        if self._measures.is_SOT_active:

            is_contactable = tracked_mask
            is_contactable_indices = np.nonzero(is_contactable)[0]

            SOT_contacts_mask = np.zeros_like(infected_mask, dtype=np.bool)
            if len(is_contactable_indices) > 0:
                for i in range(backtrack_length):

                    (
                        contact_cols,
                        contact_strengths,
                    ) = self._population.get_contacts(
                        contacted_indices,
                        is_contactable_indices,
                        return_rows=False,
                    )

                    successful_contacts_mask = (
                        self._rstate.poisson(contact_strengths) >= 1
                    )

                    successful_contacts_indices = contact_cols[
                        successful_contacts_mask
                    ]

                    if len(successful_contacts_indices) > 0:
                        SOT_contacts_mask[successful_contacts_indices] = True

            contacted_mask = np.logical_or(contacted_mask, SOT_contacts_mask)

        cond = np.logical_or(tracked_infected_mask, contacted_mask)

        return cond

    def __will_be_tested(self, data: DataDict) -> np.ndarray:
        new_quarantined = data["is_new_quarantined"]
        num_new_quarantined = new_quarantined.sum()
        new_quarantined_indices = new_quarantined.nonzero()[0]

        will_be_tested = np.ones(num_new_quarantined, dtype=np.bool)

        will_be_tested_indices = new_quarantined_indices[will_be_tested]
        cond = np.zeros(data.field_len, dtype=np.bool)
        cond[will_be_tested_indices] = True

        return cond

    def __will_test_negative(self, data: DataDict) -> np.ndarray:

        new_test_mask = self.states["is_new_tested"](data)

        infectious_mask = self.states["is_infectious"](data) & ~self.states[
            "is_new_infectious"
        ](data)

        non_testable_mask = ~self.states["is_infectious"](data)
        always_negative_mask = new_test_mask & non_testable_mask
        always_negative_indices = np.nonzero(always_negative_mask)[0]

        cond = np.zeros(data.field_len, dtype=np.bool)
        cond[always_negative_indices] = True

        infectious_to_test_mask = new_test_mask & infectious_mask
        if not np.any(infectious_to_test_mask):
            return cond

        infectious_to_test_indices = np.nonzero(infectious_to_test_mask)[0]
        infectious_dur = data["time_since_infectious"][
            infectious_to_test_indices
        ]
        correct_test_prob = self._measures.test_efficiency(infectious_dur)
        positive_test_mask = self._rstate.binomial(1, correct_test_prob)
        positive_test_mask = np.asarray(positive_test_mask, np.bool)
        false_negative_indices = infectious_to_test_indices[
            ~positive_test_mask
        ]
        cond[false_negative_indices] = True

        return cond
