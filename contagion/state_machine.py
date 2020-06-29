# -*- coding: utf-8 -*-

"""
Name: state_machine.py
Authors: Christian Haack, Stephan Meighen-Berger, Andrea Turcati
Constructs the state machine
"""

from __future__ import annotations
import abc
from collections import defaultdict
from copy import deepcopy
import functools
import logging
from typing import Callable, Union, List, Tuple, Dict, Optional

import numpy as np  # type: ignore
import networkx as nx  # type: ignore
import pandas as pd  # type: ignore

from .config import config
from .infection import Infection
from .measures import Measures
from .pdfs import PDF
from .population import Population, NetworkXPopulation

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

    def __or__(self, other: TCondition) -> Condition:
        """
        Logical and of a condition and an object of type `TCondition`

        Parameters:
            other: TCondition

        """

        def new_condition(data: DataDict):
            cond = unify_condition(other, data)

            return self(data) | cond

        return Condition(new_condition)


TCondition = Union["_State", np.ndarray, Condition, None]


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
    ) -> np.ndarray:
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
        cond = cond & self_cond
        self._state_change(data, state, cond)

        return cond

    @property
    def name(self):
        return self._name


class BooleanState(_State):
    """
    Specialization for boolean states.
    """

    @classmethod
    def from_boolean(
        cls, name: str, graph: Optional[nx.Graph] = None
    ) -> BooleanState:
        """
        Factory method for creating a state from a boolean field
        in a DataDict. The name of the state corresponds to the data field name
        in the DataDict.

        Parameters:
            name: str
            graph: Optional[nx.Graph]
                A graph object onto which the state change is recorded
        """

        def get_state(arr: np.ndarray):
            return arr

        def state_change(
            data: DataDict, state: np.ndarray, condition: np.ndarray,
        ):

            # TODO: maybe offload application of condition to state here?
            data[name][condition] = state

            if graph is not None:
                sel_nodes = np.asarray(graph.nodes)[condition]

                for i, node in enumerate(sel_nodes):
                    if isinstance(state, np.ndarray):
                        this_state = state[i]
                    else:
                        this_state = state
                    if name not in graph.nodes[node]["history"]:
                        graph.nodes[node]["history"][name] = []

                    graph.nodes[node]["history"][name].append(
                        (graph.graph["cur_tick"], this_state)
                    )

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
    def __call__(self, data: DataDict) -> np.ndarray:
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
        changed = self._state_a.change_state(data, False)

        return changed


class ChangeStateConditionalTransition(_Transition, ConditionalMixin):
    """
    Change a state where external condition is true

    Parameters:
        name: str
        state_a: Union[_State, Tuple[_State, bool]]
            State to change. Can either be a `_State`, in which case the state
            will be set to true or a Tuple[_State, bool], where the second item
            is the value the state should be set to.

    """

    _state_a: _State
    _state_a_val: bool

    def __init__(
        self,
        name: str,
        state_a: Union[_State, Tuple[_State, bool]],
        condition: TCondition = None,
        pipe_condition_mask: bool = False,
        log=False,
        *args,
        **kwargs,
    ):
        _Transition.__init__(self, name, *args, **kwargs)
        ConditionalMixin.__init__(self, condition, *args, **kwargs)
        self._log = log
        self._pipe_condition_mask = pipe_condition_mask

        if isinstance(state_a, tuple):
            self._state_a = state_a[0]
            self._state_a_val = state_a[1]
        else:
            self._state_a = state_a
            self._state_a_val = True

    @log_call
    def __call__(
            self,
            data: DataDict,
            condition_mask: Optional[np.ndarray] = None):
        cond = self.unify_condition(data)

        if condition_mask is not None:
            cond = cond & condition_mask

        if callable(self._state_a_val):
            parsed_val = self._state_a_val(data, cond & self._state_a(data))
        else:
            parsed_val = self._state_a_val

        changed = self._state_a.change_state(data, parsed_val, cond)
        if self._log:
            print("changed: ", changed.sum())
            print(self._name)
            #print("Nonzero cond ", np.nonzero(condition_mask))
            if np.any(np.nonzero(cond)[0] == 0):
                print(data["time_until_second_test"][0])
        if self._pipe_condition_mask:
            return condition_mask
        return changed


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
    def __call__(
            self,
            data: DataDict,
            condition_mask: Optional[np.ndarray] = None) -> np.ndarray:
        cond = self.unify_condition(data)
        if condition_mask is not None:
            cond = cond & condition_mask

        (~self._state_b).change_state(data, True, cond & self._state_a(data))
        changed = self._state_a.change_state(data, False, cond)

        return changed


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
    def __call__(
            self,
            data: DataDict,
            condition_mask: Optional[np.ndarray] = None) -> np.ndarray:
        cond = self.unify_condition(data)
        if condition_mask is not None:
            cond = cond & condition_mask
        state_condition = self._state_a(data)

        # Current state value
        cur_state = self._state_a.get_state_value(data)[cond & state_condition]
        changed = self._state_a.change_state(data, cur_state - 1, cond)

        return changed


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
    def __call__(
            self,
            data: DataDict,
            condition_mask: Optional[np.ndarray] = None) -> np.ndarray:
        cond = self.unify_condition(data)
        if condition_mask is not None:
            cond = cond & condition_mask
        state_condition = self._state_a(data)

        # Current state value
        cur_state = self._state_a.get_state_value(data)[cond & state_condition]
        changed = self._state_a.change_state(data, cur_state + 1, cond)

        return changed


class InitializeTimerTransition(_Transition, ConditionalMixin):
    """
    Initialize a timer state to values drawn from a PDF

    Parameters:
        name: str
        state_a: FloatState
        initialization_pdf: PDF
        condition: Optional[TCondition]
        pipe_condition_mask: bool
            This option is useful for chaining transitions. The condition mask
            in __call__ will be returned.
    """

    def __init__(
        self,
        name: str,
        state_a: FloatState,
        initialization_pdf: Optional[PDF] = None,
        condition: TCondition = None,
        pipe_condition_mask: bool = False,
        stateful_init_func=None,
        log=False,
        *args,
        **kwargs,
    ):

        _Transition.__init__(self, name, *args, **kwargs)
        ConditionalMixin.__init__(self, condition)
        self._state_a = state_a
        if initialization_pdf is None and stateful_init_func is None:
            raise ValueError(
                "Must either supply initialization_pdf or stateful_init_func")
        if initialization_pdf is not None and stateful_init_func is not None:
            raise ValueError(
                "Supply only one of initialization_pdf or stateful_init_func")

        self._initialization_pdf = initialization_pdf
        self._stateful_init_func = stateful_init_func
        self._pipe_condition_mask = pipe_condition_mask
        self._log = log

    @log_call
    def __call__(
            self,
            data: DataDict,
            condition_mask: Optional[np.ndarray] = None) -> np.ndarray:
        cond = self.unify_condition(data)
        if condition_mask is not None:
            cond = cond & condition_mask

        # Rows which are currently 0
        zero_rows = (~self._state_a(data)) & cond
        num_zero_rows = zero_rows.sum(axis=0)
        if self._initialization_pdf is not None:
            initial_vals = self._initialization_pdf.rvs(num_zero_rows)
        else:
            initial_vals = self._stateful_init_func(data, zero_rows)
        # print(len(initial_vals), (~self._state_a(data)).sum())
        changed = (~self._state_a).change_state(data, initial_vals, cond)
        if self._log:
            print(self._name)
            # print("Nonzero cond ", np.nonzero(condition_mask))
            if np.any(np.nonzero(cond)[0] == 0):
                print("Time until second ", data["time_until_second_test"][0])

        if self._pipe_condition_mask:
            return condition_mask
        return changed


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
        pipe_condition_mask: bool = False,
        *args,
        **kwargs,
    ):

        _Transition.__init__(self, name, *args, **kwargs)
        ConditionalMixin.__init__(self, condition)
        self._state_a = state_a
        self._start = start
        self._pipe_condition_mask = pipe_condition_mask

    @log_call
    def __call__(
            self,
            data: DataDict,
            condition_mask: Optional[np.ndarray] = None) -> np.ndarray:
        cond = self.unify_condition(data)
        if condition_mask is not None:
            cond = cond & condition_mask
        initial_vals = self._start  # Initializes counter at 1
        changed = (~self._state_a).change_state(data, initial_vals, cond)

        if self._pipe_condition_mask:
            return condition_mask
        return changed


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
        state_a: Union[
            _State, Tuple[_State, bool],
            Tuple[_State, Callable[[DataDict, np.ndarray], np.ndarray]]
            ],
        states_b: List[Union[_State, Tuple[_State, bool]]],
        condition: TCondition,
        pipe_condition_mask: bool = False,
        *args,
        **kwargs,
    ):

        _Transition.__init__(self, name, *args, **kwargs)
        ConditionalMixin.__init__(self, condition, *args, **kwargs)
        self._pipe_condition_mask = pipe_condition_mask

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
    def __call__(
            self,
            data: DataDict,
            condition_mask: Optional[np.ndarray] = None):
        cond = self.unify_condition(data)
        if condition_mask is not None:
            cond = cond & condition_mask

        is_in_state_a = self._state_a(data)
        cond_and_a = cond & is_in_state_a

        for state, val in zip(self._states_b, self._states_b_vals):
            if callable(val):
                parsed_val = val(data, cond_and_a)
            else:
                parsed_val = val

            (~state).change_state(data, parsed_val, cond_and_a)

        changed = self._state_a.change_state(data, self._state_a_val, cond)

        if self._pipe_condition_mask:
            return condition_mask
        return changed


class TransitionChain(ConditionalMixin):
    def __init__(
        self,
        name: str,
        transitions: List[_Transition],
        carry_condition: bool = True,
        loop_until: Optional[TCondition] = None,
        *args,
        **kwargs,
    ):
        ConditionalMixin.__init__(self, loop_until, *args, **kwargs)
        self._name = name
        self._transitions = transitions
        self._carry_condition = carry_condition

    def __call__(self, data: DataDict) -> np.ndarray:
        changed = None

        loopcnt = 0
        while True:
            for transition in self._transitions:
                try:
                    if self._carry_condition:
                        changed = transition(data, changed)
                    else:
                        transition(data)
                except Exception as e:
                    print("Caught exception in transition: ", transition.name)
                    raise e
            cond = self.unify_condition(data)
            if ~np.any(cond) or self._condition is None:
                break
            loopcnt += 1
            #print(loopcnt, cond.nonzero())

    @property
    def name(self):
        return self._name


class StatCollector(object, metaclass=abc.ABCMeta):
    """
    Convenience class for collecting statistics

    Parameters:
        data_fields: List[str]
            List of data field names to track
    """

    _statistics: Dict[str, List[float]]

    def __init__(
        self,
        data_fields: List[str],
        cond_fields: Optional[List[Tuple[str, str, bool]]],
    ):
        self._data_fields = data_fields
        self._cond_fields = cond_fields
        self._statistics = defaultdict(list)
        self._recovered_old = None

    def __call__(self, data: DataDict, inf_trace=None):
        for field in self._data_fields:
            self._statistics[field].append(data[field].sum())
        if self._cond_fields is not None:
            for (field, field2, state) in self._cond_fields:
                cond = data[field] & (data[field2] == state)
                self._statistics[field + "_" + field2 + "_" + str(state)].append(cond.sum())

        # Re
        if inf_trace is not None:
            if self._recovered_old is None:
                self._statistics["Re"].append(0)
            else:
                new_recoveries = (~self._recovered_old) & data["is_recovered"]
                num_new_rec = new_recoveries.sum()
                if num_new_rec == 0:
                    re = 0
                else:
                    new_rec_ids = np.nonzero(new_recoveries)[0]
                    inf_hist = np.atleast_2d(np.squeeze(np.hstack(inf_trace)))

                    if len(inf_hist) == 0:
                        re = 0
                    else:
                        num_infected = 0
                        for new_rec_id in new_rec_ids:
                            mask = inf_hist[:, 0] == new_rec_id
                            num_infected += mask.sum()
                        re = num_infected / num_new_rec

                self._statistics["Re"].append(re)
            self._recovered_old = np.array(data["is_recovered"], copy=True)

    def __getitem__(self, key):
        return self._statistics[key]

    def __setitem__(self, key, value):
        self._statistics[key] = value

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

    # _graph_node_hist: Optional[List[Dict[int, Any]]]

    def __init__(
        self,
        data: Union[pd.DataFrame, DataDict],
        stat_collector: Optional[StatCollector],
        trace_states: bool = False,
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
            self._trace_infectable = []
        else:
            self._trace_contacts = None
            self._trace_infection = None
            self._trace_infectable = None

        self._cur_tick = 0
        self._trace_states = trace_states
        self._traced_states = []

    @property
    @abc.abstractmethod
    def transitions(self) -> List[_Transition]:
        pass

    @property
    @abc.abstractmethod
    def states(self) -> List[_State]:
        pass

    def tick(self) -> bool:
        """
        Perform all transitions
        """
        for transition in self.transitions:
            try:
                transition(self._data)
            except Exception as e:
                print("Caught exception in transition: ", transition.name)
                raise(e)
        if self._stat_collector is not None:
            self._stat_collector(self._data, self._trace_infection)
        self._cur_tick += 1

        if self._trace_states:
            self._traced_states.append(deepcopy(self._data))
        return False

        """
        if isinstance(self._population, NetworkXPopulation):
            g = self._population._graph
            if self._graph_node_hist is None:
                self._graph_node_hist = []

            self._graph_node_hist.append(
                nx.get_node_attributes(g))
            # track graph attribute changes
        """

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

        graph = None
        if isinstance(self._population, NetworkXPopulation):
            graph = self._population._graph
            graph.graph["cur_tick"] = 0

        # Boolean states
        boolean_state_names = [
            "is_patient_zero",
            "is_latent",
            "will_have_symptoms",
            "is_symptomatic",
            "is_infectious",
            "is_hospitalized",
            "is_new_hospitalized",
            "will_be_hospitalized",
            "is_dead",
            "is_removed",
            "is_recovering",
            "is_recovered",
            "will_die",
            "is_reported",
            "was_infected",
            "is_index_case"
        ]

        if self._measures.contact_tracing:
            boolean_state_names.append("is_tracked")
        if self._measures.quarantine:
            boolean_state_names.append("is_quarantined")
            boolean_state_names.append("is_reported")
        if self._measures.testing:
            boolean_state_names += [
                "is_tested",
                "has_first_test_result",
                "has_second_test_result",
                "is_tested_second",
                "is_tested_negative",
                "is_tested_negative_second",
                "is_tested_positive",
                "is_tested_positive_second",
                "will_test_negative",
                "is_tracable",
                "is_rnd_tested"
                ]

        tracked_boolean_state_names = [
            "is_infected",
        ]

        boolean_states = {
            name: BooleanState.from_boolean(name, None)
            for name in boolean_state_names
        }

        for name in tracked_boolean_state_names:
            boolean_states[name] = BooleanState.from_boolean(
                name,
                graph if config["general"]["track graph history"] else None,
            )

        # Timer states
        timer_state_names = [
            "latent_duration",
            "time_until_symptoms",
            "infectious_duration",
            "hospitalization_duration",
            "recovery_time",
            "time_until_hospitalization",
            "time_until_death",
        ]

        # Counter states
        counter_state_names = [
            "time_since_infectious",
        ]

        if self._measures.quarantine:
            timer_state_names.append("quarantine_duration")
        if self._measures.testing:
            timer_state_names.append("time_until_test")
            timer_state_names.append("time_until_test_result")
            timer_state_names.append("time_until_second_test")
            timer_state_names.append("time_until_second_test_result")
            counter_state_names.append("time_since_last_test_result")

        timer_states = {
            name: FloatState.from_timer(name) for name in timer_state_names
        }

        counter_states = {
            name: FloatState.from_counter(name) for name in counter_state_names
        }

        self._states = {}
        self._states.update(boolean_states)
        self._states.update(timer_states)
        self._states.update(counter_states)

        # Transitions
        self._transitions = []

        # Healthy - latent
        # Transition from not-infected to:
        #   - is_latent
        #   - is_infected
        # if the is infectable condition is true

        # Condition that stores the new infections from this tick
        if self._measures.quarantine:
            infected_condition = (
                Condition(self.__get_new_infections)
                & Condition.from_state(~boolean_states["is_removed"])
                & Condition.from_state(~boolean_states["is_quarantined"])
            )
        else:
            infected_condition = Condition(
                self.__get_new_infections
            ) & Condition.from_state(~boolean_states["is_removed"])
        infected_condition = (
            infected_condition |
            Condition.from_state(boolean_states["is_patient_zero"])
        )
        self._transitions.append(
            TransitionChain(
                "healthy_latent_timer",
                [
                    ChangeStateConditionalTransition(
                        "healthy_is_infected",
                        ~boolean_states["is_infected"],
                        infected_condition
                    ),
                    ChangeStateConditionalTransition(
                        "was_infected",
                        ~boolean_states["was_infected"],
                        pipe_condition_mask=True
                    ),
                    ChangeStateConditionalTransition(
                        "will_have_symptoms",
                        (
                            ~boolean_states["will_have_symptoms"],
                            self.__will_have_symptoms
                        ),
                        pipe_condition_mask=True
                    ),
                    ChangeStateConditionalTransition(
                        "is_latent",
                        ~boolean_states["is_latent"],
                    ),

                    InitializeTimerTransition(
                        "init_time_until_symptoms",
                        timer_states["time_until_symptoms"],
                        self._infection.incubation_duration,
                        pipe_condition_mask=True,
                    ),

                    InitializeTimerTransition(
                        "init_latent_duration",
                        timer_states["latent_duration"],
                        stateful_init_func=self.__correlated_latent,
                        # self._infection.latent_duration,
                        pipe_condition_mask=True,
                        log=False
                    )
                ]
            )
        )

        # Latent - Infectious
        infectious_cond = Condition.from_state(
            ~timer_states["latent_duration"]
        )

        self._transitions.append(
            TransitionChain(
                "latent_infectious_timer",
                [
                    ConditionalTransition(
                        "latent_infectious",
                        boolean_states["is_latent"],
                        boolean_states["is_infectious"],
                        infectious_cond,
                    ),

                    InitializeTimerTransition(
                        "init_infectious_duration",
                        timer_states["infectious_duration"],
                        self._infection.infectious_duration,
                        pipe_condition_mask=True
                    ),

                    InitializeCounterTransition(
                        "init_time_since_infectious",
                        counter_states["time_since_infectious"],
                    )
                ]
            )
        )

        # No Symptoms - Symptomatic
        symptomatic_cond = Condition.from_state(
            ~timer_states["time_until_symptoms"]
        )

        self._transitions.append(
            TransitionChain(
                "no_symptoms_symptomatic_hospit_death",
                [
                    ConditionalTransition(
                        "no_symptoms_symptomatic",
                        boolean_states["will_have_symptoms"],
                        boolean_states["is_symptomatic"],
                        symptomatic_cond,
                    ),

                    ChangeStateConditionalTransition(
                        "will_be_hospitalized",
                        (
                            ~boolean_states["will_be_hospitalized"],
                            self.__will_be_hospitalized
                        ),
                        None
                    ),

                    InitializeTimerTransition(
                        "init_time_until_hospitalization",
                        timer_states["time_until_hospitalization"],
                        self._infection.time_until_hospitalization,
                        pipe_condition_mask=True,
                    ),

                    ChangeStateConditionalTransition(
                        "will_die",
                        (
                            ~boolean_states["will_die"],
                            self.__will_die
                        )
                    ),

                    InitializeTimerTransition(
                        "init_time_until_death",
                        timer_states["time_until_death"],
                        self._infection.time_incubation_death,
                    ),
                ]
            )
        )

        # infectious - recovering
        # Transition from is_infectious to:
        #   -is_recovering
        #   -is_removed
        # where the normal_recovery_condition is True
        normal_recovery_condition = (
            Condition.from_state(~(boolean_states["is_hospitalized"]))
            & Condition.from_state(~timer_states["infectious_duration"])
        )

        if self._measures.quarantine:
            self._transitions.append(
                TransitionChain(
                    "infectious_recovering_timer",
                    [
                        MultiStateConditionalTransition(
                            "infectious_recovering",
                            boolean_states["is_infectious"],
                            [
                                boolean_states["is_recovering"],
                                boolean_states["is_removed"],
                                (~boolean_states["is_symptomatic"], False),
                                (~boolean_states["will_have_symptoms"], False),
                                #(~boolean_states["is_reported"], False),
                                (~boolean_states["is_quarantined"], False),
                                (~counter_states["time_since_infectious"],
                                 -np.inf),
                            ],
                            normal_recovery_condition,
                        ),

                        InitializeTimerTransition(
                            "init_recovery_time",
                            timer_states["recovery_time"],
                            self._infection.recovery_time,
                        ),
                    ]
                )
            )
        else:
            self._transitions.append(
                TransitionChain(
                    "infectious_recovering_timer",
                    [
                        MultiStateConditionalTransition(
                            "infectious_recovering",
                            boolean_states["is_infectious"],
                            [
                                boolean_states["is_recovering"],
                                boolean_states["is_removed"],
                                (~boolean_states["is_symptomatic"], False),
                                (~boolean_states["will_have_symptoms"], False),
                                #(~boolean_states["is_reported"], False),
                                (~counter_states["time_since_infectious"],
                                 -np.inf),
                            ],
                            normal_recovery_condition,
                        ),

                        InitializeTimerTransition(
                            "init_recovery_time",
                            timer_states["recovery_time"],
                            self._infection.recovery_time,
                        ),
                    ]
                )
            )

        # recovering - recovered
        # Transition from is_recovering to:
        #   -is_recovered
        #   -not is_infected
        # where the recovery_time timer is <= 0
        recovered_cond = Condition.from_state(
            ~timer_states["recovery_time"]
        )
        if self._measures.testing:
            self._transitions.append(

                MultiStateConditionalTransition(
                    "recovering_recovered",
                    boolean_states["is_recovering"],
                    [
                        boolean_states["is_recovered"],
                        (~boolean_states["is_infected"], False),
                        (~boolean_states["is_reported"], False),
                    ],
                    recovered_cond,
                )
            )
        else:
            self._transitions.append(

                MultiStateConditionalTransition(
                    "recovering_recovered",
                    boolean_states["is_recovering"],
                    [
                        boolean_states["is_recovered"],
                        (~boolean_states["is_infected"], False),
                    ],
                    recovered_cond,
                )
            )

        # will_be_hospitalized - hospitalized
        # Transition from will_be_hospitalized to:
        #   -is_hospitalized
        #   -is_new_hospitalized
        #   -is_removed
        #   -not is_recovering
        # where the time_until_hospitalization timer is <= 0
        hospit_cond = Condition.from_state(
            ~timer_states["time_until_hospitalization"]
        )
        if self._measures.testing & self._measures.quarantine:
            self._transitions.append(
                TransitionChain(
                    "will_be_hospitalized_hospitalized_timer",
                    [
                        MultiStateConditionalTransition(
                            "will_be_hospitalized_hospitalized",
                            boolean_states["will_be_hospitalized"],
                            [
                                boolean_states["is_hospitalized"],
                                boolean_states["is_removed"],
                                (~boolean_states["will_test_negative"], False),
                                (~boolean_states["is_tested"], False),
                                boolean_states["is_tested_positive"],
                                (~boolean_states["is_quarantined"], False),
                            ],
                            hospit_cond,
                        ),

                        InitializeTimerTransition(
                            "init_hospitalization_duration",
                            timer_states["hospitalization_duration"],
                            self._infection.hospitalization_duration,
                        ),

                    ]
                )
            )
        else:
            self._transitions.append(
                TransitionChain(
                    "will_be_hospitalized_hospitalized_timer",
                    [
                        MultiStateConditionalTransition(
                            "will_be_hospitalized_hospitalized",
                            boolean_states["will_be_hospitalized"],
                            [
                                boolean_states["is_hospitalized"],
                                boolean_states["is_removed"],
                            ],
                            hospit_cond,
                        ),

                        InitializeTimerTransition(
                            "init_hospitalization_duration",
                            timer_states["hospitalization_duration"],
                            self._infection.hospitalization_duration,
                        ),
                    ]
                )
            )

        # hospitalized - recovered
        hospit_recovery_condition = (
            Condition.from_state(~timer_states["hospitalization_duration"])
            & Condition.from_state(~boolean_states["will_die"])
        )
        self._transitions.append(
            MultiStateConditionalTransition(
                "hospitalized_recovered",
                boolean_states["is_hospitalized"],
                [
                    boolean_states["is_recovered"],
                    (~boolean_states["is_infected"], False),
                    (~boolean_states["is_symptomatic"], False),
                ],
                hospit_recovery_condition,
            )
        )

        # will_die - is_dead
        # Transition from will_die to:
        #   -is_dead
        #   -is_removed
        #   -not is_infected
        #   -not is_hospitalized
        #   -not is_new_hospitalized
        # where the time_until_death timer is <= 0
        is_dead_cond = Condition.from_state(
            ~timer_states["time_until_death"]
        )
        self._transitions.append(
            MultiStateConditionalTransition(
                "will_die_is_dead",
                boolean_states["will_die"],
                [
                    boolean_states["is_dead"],
                    boolean_states["is_removed"],
                    (~boolean_states["is_infectious"], False),
                    (~boolean_states["is_infected"], False),
                    (~boolean_states["is_hospitalized"], False),
                    (~boolean_states["is_symptomatic"], False),
                    (~boolean_states["will_have_symptoms"], False),
                ],
                is_dead_cond,
            )
        )

        if self._measures.random_testing:
            contacts_traced_cond = Condition(
                    self.__tracing_active)
            rnd_test_cond = Condition(
                self.__takes_random_test) & contacts_traced_cond

            test_result_cond = Condition.from_state(
                ~timer_states["time_until_test_result"]
            ) & Condition.from_state(
               boolean_states["is_rnd_tested"])
            test_negative_cond = Condition.from_state(
                boolean_states["will_test_negative"]
            ) & Condition.from_state(
               boolean_states["is_rnd_tested"])
            test_positive_cond = Condition.from_state(
                ~boolean_states["will_test_negative"]
            ) & Condition.from_state(
               boolean_states["is_rnd_tested"])

            self._transitions.append(
                TransitionChain(
                    "rnd_test",
                    [
                        ChangeStateConditionalTransition(
                            "is_rnd_tested",
                            ~boolean_states["is_rnd_tested"],
                            rnd_test_cond
                        ),

                        InitializeTimerTransition(
                            "init_time_until_test_result",
                            timer_states["time_until_test_result"],
                            self._measures.time_until_test_result,
                            pipe_condition_mask=True
                        ),

                        ChangeStateConditionalTransition(
                            "will_test_negative",
                            (
                                ~boolean_states["will_test_negative"],
                                self.__will_test_negative
                            ),
                            pipe_condition_mask=True
                        ),
                    ]
                )
            )
            self._transitions.append(
                TransitionChain(
                    "tested_positive_rnd_timer",
                    [
                        MultiStateConditionalTransition(
                            "tested_positive",
                            boolean_states["is_rnd_tested"],
                            [
                                boolean_states["is_tested_positive"],
                                boolean_states["is_reported"],
                                boolean_states["is_index_case"],
                                boolean_states["is_quarantined"],
                            ],
                            test_positive_cond & test_result_cond
                        ),

                        ChangeStateConditionalTransition(
                            "reset_time_since_last_test_result",
                            (counter_states["time_since_last_test_result"],
                             -np.inf
                             ),
                            pipe_condition_mask=True),

                        InitializeCounterTransition(
                            "init_time_since_last_test_result",
                            counter_states["time_since_last_test_result"],
                            pipe_condition_mask=True
                        ),
                    ]
                )
            )
            self._transitions.append(
                TransitionChain(
                    "tested_negative_rnd",
                    [
                        ChangeStateConditionalTransition(
                            "tested_negative",
                            (boolean_states["is_rnd_tested"], False),
                            test_negative_cond & test_result_cond
                        ),

                        ChangeStateConditionalTransition(
                            "reset_will_test_negative",
                            (boolean_states["will_test_negative"], False),
                            pipe_condition_mask=True
                        ),

                        ChangeStateConditionalTransition(
                            "reset_time_since_last_test_result",
                            (counter_states["time_since_last_test_result"],
                             -np.inf
                             ),
                            pipe_condition_mask=True),
                    ]
                )
            )

        if self._measures.quarantine:
            # Susceptible - Quarantine
            # transition if:
            #  - is_infected
            #  - is_tracked

            quarantine_condition = (
                Condition.from_state(
                        boolean_states["is_symptomatic"]) &
                Condition.from_state(
                    ~(boolean_states["is_removed"])
                ) &
                Condition.from_state(
                    ~(boolean_states["is_quarantined"])
                )
            )

            is_tracable_cond = Condition.from_state(
                boolean_states["is_tracable"])

            self._transitions.append(
                TransitionChain(
                    "symptomatic_indexcase",
                    [
                        ChangeStateConditionalTransition(
                            "reported",
                            ~boolean_states["is_reported"],
                            quarantine_condition & Condition(self.__tracing_active),

                        ),
                        ChangeStateConditionalTransition(
                            "index_case",
                            ~boolean_states["is_index_case"],
                        )
                    ]
                )
            )

            if self._measures.testing:

                self._transitions.append(
                    TransitionChain(
                        "quarantined_tested_timer",
                        [
                            ChangeStateConditionalTransition(
                                "quarantined_symptomatic",
                                ~boolean_states["is_quarantined"],
                                quarantine_condition
                            ),
                            ChangeStateConditionalTransition(
                                "is_tested_positive",
                                ~boolean_states["is_tested_positive"],
                                pipe_condition_mask=True
                            ),
                            ChangeStateConditionalTransition(
                                "reset_time_since_last_test_result",
                                (counter_states["time_since_last_test_result"],
                                 -np.inf),
                                pipe_condition_mask=True),
                            InitializeCounterTransition(
                                "init_time_since_last_test_result",
                                counter_states["time_since_last_test_result"],
                                pipe_condition_mask=True
                            ),
                        ]
                    )
                )

                has_been_traced_cond = (
                    Condition(self.__quarantined_traced) &
                    Condition.from_state(
                        ~boolean_states["is_removed"]) &
                    Condition.from_state(
                        ~boolean_states["is_quarantined"]) &
                    is_tracable_cond
                )
                contacts_traced_cond = Condition(
                    self.__tracing_active)

                loop_condition = (
                    Condition.from_state(
                        boolean_states["is_reported"])
                    & contacts_traced_cond
                )

                test_result_cond = (
                    Condition.from_state(
                        ~timer_states["time_until_test_result"]
                    ) &
                    Condition.from_state(
                        boolean_states["is_quarantined"]
                    ) &
                    Condition.from_state(
                        boolean_states["is_tested"]
                    ) &
                    Condition.from_state(
                        ~boolean_states["has_first_test_result"]
                    )
                )

                test_negative_cond = Condition.from_state(
                    boolean_states["will_test_negative"]
                ) & Condition.from_state(
                   boolean_states["is_quarantined"])

                test_positive_cond = Condition.from_state(
                    ~boolean_states["will_test_negative"]
                ) & Condition.from_state(
                   boolean_states["is_quarantined"])

                time_until_second_test_cond = (
                    Condition.from_state(
                        ~timer_states["time_until_second_test"]
                    ) &
                    Condition.from_state(
                        boolean_states["is_tested_negative"]
                    ) &
                    Condition.from_state(
                        boolean_states["is_quarantined"]
                    ) &
                    Condition.from_state(
                        ~boolean_states["is_tested_second"]
                    )
                )

                second_test_result_cond = (
                    Condition.from_state(
                        ~timer_states["time_until_second_test_result"]
                    ) &
                    Condition.from_state(
                        boolean_states["is_tested_second"]
                    ) &
                    Condition.from_state(
                        boolean_states["is_quarantined"]
                    ) &
                    Condition.from_state(
                        ~boolean_states["has_second_test_result"]
                    )
                )

                self._transitions.append(
                    TransitionChain(
                        "tracing_loop",
                        [
                            TransitionChain(
                                "traced_quarantine",
                                [
                                    ChangeStateConditionalTransition(
                                        "quarantined",
                                        ~boolean_states["is_quarantined"],
                                        has_been_traced_cond
                                    ),

                                    ChangeStateConditionalTransition(
                                        "is_tested",
                                        ~boolean_states["is_tested"],
                                        pipe_condition_mask=True
                                    ),
                                    ChangeStateConditionalTransition(
                                        "reset_time_since_last_test_result",
                                        (counter_states["time_since_last_test_result"],
                                         -np.inf),
                                        pipe_condition_mask=True),

                                    InitializeTimerTransition(
                                        "init_time_until_test_result",
                                        timer_states["time_until_test_result"],
                                        self._measures.time_until_test_result,
                                        pipe_condition_mask=True
                                    ),

                                    ChangeStateConditionalTransition(
                                        "will_test_negative",
                                        (
                                            ~boolean_states["will_test_negative"],
                                            self.__will_test_negative
                                        ),
                                        pipe_condition_mask=True
                                    ),
                                    InitializeCounterTransition(
                                        "init_time_since_last_test_result",
                                        counter_states["time_since_last_test_result"],
                                        pipe_condition_mask=True
                                    ),
                                ]
                            ),
                            ChangeStateConditionalTransition(
                                "reported_off",
                                (boolean_states["is_reported"], False),
                                contacts_traced_cond,
                            ),
                            TransitionChain(
                                "tested_positive_timer",
                                [
                                    ChangeStateConditionalTransition(
                                        "is_tested_positive",
                                        ~boolean_states["is_tested_positive"],
                                        test_positive_cond & test_result_cond,
                                    ),
                                    ChangeStateConditionalTransition(
                                        "has_first_result",
                                        ~boolean_states["has_first_test_result"],
                                        pipe_condition_mask=True
                                    ),
                                    ChangeStateConditionalTransition(
                                        "is_reported",
                                        ~boolean_states["is_reported"],
                                        pipe_condition_mask=True
                                    ),
                                    ChangeStateConditionalTransition(
                                        "is_index_case",
                                        ~boolean_states["is_index_case"],
                                        pipe_condition_mask=True
                                    ),

                                    ChangeStateConditionalTransition(
                                        "reset_time_since_last_test_result",
                                        (counter_states["time_since_last_test_result"],
                                         -np.inf
                                         ),
                                        pipe_condition_mask=True),

                                    InitializeCounterTransition(
                                        "init_time_since_last_test_result",
                                        counter_states["time_since_last_test_result"],
                                        pipe_condition_mask=True
                                    ),
                                ]
                            ),
                            TransitionChain(
                                "tested_negative_retest_timer",
                                [
                                    ChangeStateConditionalTransition(
                                        "tested_negative",
                                        ~boolean_states["is_tested_negative"],
                                        test_negative_cond & test_result_cond,
                                    ),
                                    ChangeStateConditionalTransition(
                                        "has_first_result",
                                        ~boolean_states["has_first_test_result"],
                                        pipe_condition_mask=True,
                                    ),
                                    ChangeStateConditionalTransition(
                                        "reset_will_test_negative",
                                        (boolean_states["will_test_negative"], False),
                                        pipe_condition_mask=True
                                    ),
                                    ChangeStateConditionalTransition(
                                        "reset_time_until_second_test",
                                        (timer_states["time_until_second_test"],
                                         0
                                         ),
                                        pipe_condition_mask=True
                                    ),

                                    InitializeTimerTransition(
                                        "init_time_until_second_test",
                                        timer_states["time_until_second_test"],
                                        self._measures.time_until_second_test,
                                        pipe_condition_mask=True,
                                        log=False
                                    ),

                                    # ChangeStateConditionalTransition(
                                    #     "reset_time_since_last_test_result",
                                    #     (counter_states["time_since_last_test_result"],
                                    #      -np.inf
                                    #      ),
                                    #     pipe_condition_mask=True),

                                    # InitializeCounterTransition(
                                    #     "init_time_since_last_test_result",
                                    #     counter_states["time_since_last_test_result"],
                                    #     pipe_condition_mask=True
                                    # ),

                                ]
                            ),
                            TransitionChain(
                                "is_tested_second",
                                [
                                    ChangeStateConditionalTransition(
                                        "is_tested_again",
                                        ~boolean_states["is_tested_second"],
                                        time_until_second_test_cond,
                                        log=False
                                    ),

                                    ChangeStateConditionalTransition(
                                        "will_test_negative",
                                        (
                                            ~boolean_states["will_test_negative"],
                                            self.__will_test_negative
                                        ),
                                        pipe_condition_mask=True
                                    ),
                                    ChangeStateConditionalTransition(
                                        "reset_time_until_second_test_result",
                                        (timer_states["time_until_second_test_result"],
                                         0
                                         ),
                                        pipe_condition_mask=True
                                    ),

                                    InitializeTimerTransition(
                                        "init_time_until_second_test_result",
                                        timer_states["time_until_second_test_result"],
                                        self._measures.time_until_second_test_result,
                                    )
                                ]
                            ),
                            TransitionChain(
                                "tested_positive_second_timer",
                                [
                                    ChangeStateConditionalTransition(
                                        "is_tested_positive_second",
                                        ~boolean_states["is_tested_positive_second"],
                                        test_positive_cond & second_test_result_cond,
                                    ),
                                    ChangeStateConditionalTransition(
                                        "has_second_result",
                                        ~boolean_states["has_second_test_result"],
                                        pipe_condition_mask=True
                                    ),
                                    ChangeStateConditionalTransition(
                                        "is_reported",
                                        ~boolean_states["is_reported"],
                                        pipe_condition_mask=True
                                    ),
                                    ChangeStateConditionalTransition(
                                        "is_index_case",
                                        ~boolean_states["is_index_case"],
                                        pipe_condition_mask=True
                                    ),
                                    ChangeStateConditionalTransition(
                                        "reset_time_since_last_test_result",
                                        (counter_states["time_since_last_test_result"],
                                         -np.inf
                                         ),
                                        pipe_condition_mask=True
                                    ),

                                    InitializeCounterTransition(
                                        "init_time_since_last_test_result",
                                        counter_states["time_since_last_test_result"],
                                        pipe_condition_mask=True
                                    ),
                                ]
                            ),
                            TransitionChain(
                                "tested_negative_second_timer",
                                [
                                    ChangeStateConditionalTransition(
                                        "tested_negative_second",
                                        ~boolean_states["is_tested_negative_second"],
                                        test_negative_cond & second_test_result_cond
                                    ),
                                    ChangeStateConditionalTransition(
                                        "has_second_result",
                                        ~boolean_states["has_second_test_result"],
                                        pipe_condition_mask=True
                                    ),
                                ]
                            )
                        ],
                        carry_condition=False,
                        loop_until=loop_condition
                    )
                )

                # Quarantine release

                free_condition = (
                    (
                        Condition(self.__quarantine_test_free) &
                        Condition.from_state(
                            boolean_states["is_quarantined"]
                        )
                    ) |
                    Condition.from_state(
                        boolean_states["is_tested_negative_second"]
                    )
                )

                reset_states = [
                    "is_tested_positive",
                    "is_tested_negative",
                    "is_tested_positive_second",
                    "is_tested_negative_second",
                    "has_first_test_result",
                    "has_second_test_result",
                    "will_test_negative",
                    "is_tested",
                    "is_tested_second",
                    ]

                resetters = []
                for rs in reset_states:
                    resetters.append(
                        ChangeStateConditionalTransition(
                            "reset_"+rs,
                            (boolean_states[rs], False),
                            pipe_condition_mask=True
                        )
                    )

                self._transitions.append(
                    TransitionChain(
                        "quarantine_test_free",
                        [
                            ChangeStateConditionalTransition(
                                "quarantine_false",
                                (boolean_states["is_quarantined"], False),
                                free_condition
                            ),
                            ChangeStateConditionalTransition(
                                "reset_time_since_last_test_result",
                                (counter_states["time_since_last_test_result"],
                                 -np.inf
                                 ),
                                pipe_condition_mask=True
                            ),
                            ChangeStateConditionalTransition(
                                "reset_time_until_rest_result",
                                (timer_states["time_until_test_result"],
                                 0
                                 ),
                                pipe_condition_mask=True
                            ),
                            ChangeStateConditionalTransition(
                                "reset_time_until_second_test",
                                (timer_states["time_until_second_test"],
                                 0
                                 ),
                                pipe_condition_mask=True
                            ),
                            ChangeStateConditionalTransition(
                                "reset_time_until_second_test_result",
                                (timer_states["time_until_second_test_result"],
                                 0
                                 ),
                                pipe_condition_mask=True
                            ),

                        ] + resetters
                    )
                )

            else:
                self._transitions.append(
                    TransitionChain(
                        "quarantined_timer",
                        [
                            ChangeStateConditionalTransition(
                                "is_quarantined",
                                ~boolean_states["is_quarantined"],
                                quarantine_condition,
                            ),

                            InitializeTimerTransition(
                                "init_quarantine_duration",
                                timer_states["quarantine_duration"],
                                self._measures.quarantine_duration,
                            ),
                        ]
                    )
                )

                # Quarantine release

                free_condition = Condition.from_state(
                    timer_states["quarantine_duration"])

                self._transitions.append(
                    ChangeStateConditionalTransition(
                        "quarantine_free",
                        (boolean_states["is_quarantined"], False),
                        free_condition
                    )
                )

        for state in timer_states:
            self._transitions.append(
                DecreaseTimerTransition(
                    "decrease_{}".format(state),
                    timer_states[state],
                )
            )

        for state in counter_states:
            self._transitions.append(
                IncreaseTimerTransition(
                    "increase_{}".format(state),
                    counter_states[state],
                )
            )

    def tick(self) -> bool:
        """
        Perform all transitions
        """

        super().tick()

        if isinstance(self._population, NetworkXPopulation):
            self._population._graph.graph["cur_tick"] += 1

        if np.sum(self._data["is_infected"]) == 0:
            # Early stopping
            return True
        return False

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

        infectious_mask = self.states["is_infectious"](data)


        # Only infectious non removed, non-quarantined can infect others
        removed_mask = self.states["is_removed"](data)
        hospitalized_mask = self.states["is_hospitalized"](data)
        infectious_mask = (
            infectious_mask & (~removed_mask) & (~hospitalized_mask)
        )

        if self._measures.quarantine:
            quarantined_mask = self.states["is_quarantined"](data)
            infectious_mask = infectious_mask & (~quarantined_mask)

        infectious_indices = np.nonzero(infectious_mask)[0]

        if len(infectious_indices) == 0:
            self._stat_collector["contacts"].append(0)
            self._stat_collector["contacts_per_person"].append(0)
            if config["general"]["trace spread"]:
                #self._trace_contacts.append(np.empty((1, 0, 2)))
                self._trace_contacts.append(defaultdict(list))
                self._trace_infection.append(np.empty((1, 0, 2), dtype=np.int))
            return np.zeros_like(infectious_mask, dtype=np.bool)
        healthy_mask = ~self.states["is_infected"](data)
        is_infectable = healthy_mask & (~removed_mask)
        if self._measures.quarantine:
            is_infectable = is_infectable & (~quarantined_mask)
        is_infectable_indices = np.nonzero(is_infectable)[0]

        if config["general"]["trace spread"]:
            if self._measures.quarantine:
                avail = ~quarantined_mask
            else:
                avail = np.ones(data.field_len, dtype=np.bool)
            avail_ind = np.nonzero(avail)[0]
            self._trace_infectable.append(avail_ind)

        if len(is_infectable_indices) == 0:
            self._stat_collector["contacts"].append(0)
            self._stat_collector["contacts_per_person"].append(0)
            if config["general"]["trace spread"]:
                #self._trace_contacts.append(np.empty((1, 0, 2)))
                self._trace_contacts.append(defaultdict(list))
                self._trace_infection.append(np.empty((1, 0, 2)), dtype=np.int)
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

        # we are just interested in the columns, ie. only the
        # ids of the people contacted by the infected.
        # Note, that contacted ids can appear multiple times
        # if a person is successfully contacted by multiple people.

        """
        if contact_strengths is None:
            successful_contacts_indices = contact_cols
            successful_contactee_indices = contact_rows

        else:
            n_contacts = self._rstate.poisson(contact_strengths)
            successful_contacts_mask = (
                 n_contacts >= 1
            )

            # check if the successful contacts are quarantined

            # we are just interested in the columns, ie. only the
            # ids of the people contacted by the infected.
            # Note, that contacted ids can appear multiple times
            # if a person is successfully contacted by multiple people.
            successful_contacts_indices = (
                contact_cols[successful_contacts_mask]
            )
            successful_contactee_indices = (
                contact_rows[successful_contacts_mask]
            )

            successful_contacts_strength = (
                n_contacts[successful_contacts_mask])
        """

        successful_contacts_indices = contact_cols
        successful_contactee_indices = contact_rows
        successful_contacts_strength = contact_strengths
        # Calculate infection probability for all contacts
        # The duration of each infection
        infectious_dur = data["time_since_infectious"][
            successful_contactee_indices
        ]
        """
        infectious_dur += self._rstate.uniform(
            -0.5, 0.5, size=len(infectious_dur))
        """
        infection_prob = self._infection.pdf_infection_prob.pdf(
            infectious_dur)
        # An infection is successful if the bernoulli outcome
        # based on the infection probability is 1

        newly_infected_mask = self._rstate.binomial(
           successful_contacts_strength.astype(np.int32),
           infection_prob) >= 1
        newly_infected_mask = np.asarray(newly_infected_mask, bool)
        # Get the indices for the newly infected
        newly_infected_indices = successful_contacts_indices[
            newly_infected_mask
        ]

        if config["general"]["trace spread"]:
            # app starts at day 1
            # if not self._measures.measures_active:
            #    self._trace_contacts.append(defaultdict(set))
            #else:

            cdict = defaultdict(list)
            for contact, contactee in zip(
                    successful_contacts_indices,
                    successful_contactee_indices):
                cdict[contactee].append(contact)
            self._trace_contacts.append(cdict)
            """
            self._trace_contacts.append(
                 np.dstack((
                    successful_contactee_indices,
                    successful_contacts_indices)
                 )
            )
            """

        num_succesful_contacts = np.sum(successful_contacts_strength)
        num_succesful_contactees = len(np.unique(successful_contactee_indices))
        self._stat_collector["contacts"].append(num_succesful_contacts)

        if num_succesful_contactees > 0:
            c_per_p = num_succesful_contacts / num_succesful_contactees
        else:
            c_per_p = 0
        self._stat_collector["contacts_per_person"].append(
            c_per_p
            )
        newly_infectee_indices = successful_contactee_indices[
            newly_infected_mask
        ]

        if config["general"]["trace spread"]:

            self._trace_infection.append(
                np.dstack((newly_infectee_indices, newly_infected_indices))
            )

        # There might be multiple successfull infections per person
        # from different infected people
        newly_infected_indices, uq_ind = np.unique(
            newly_infected_indices, return_index=True
        )

        # If multiple sucessful interacions pick the first
        newly_infectee_indices = newly_infectee_indices[uq_ind]
        if config["measures"]["contact tracing"] and len(newly_infected_indices) > 0:
            if self._measures.backtrack_length > 0:
                # generate contacts to fill up backtrace
                bt_len = min(
                        len(self._trace_contacts),
                        self._measures.backtrack_length)
                if bt_len > 0:
                    bt_days = self._trace_contacts[-bt_len:]
                    inf_at_day = self._trace_infectable[-bt_len:]
                    tot_cont = 0
                    for bt_day, inf_day in zip(bt_days, inf_at_day):
                        (
                            contact_cols,
                            _,
                            contact_rows,
                        ) = self._population.get_contacts(
                            newly_infected_indices,
                            inf_day,
                            return_rows=True)
                        tot_cont += len(contact_cols)
                        for contact, contactee in zip(
                                contact_cols,
                                contact_rows):
                            bt_day[contactee].append(contact)

        if isinstance(self._population, NetworkXPopulation):
            # update graph history
            g = self._population._graph
            for ni, ni_by in zip(
                newly_infected_indices, newly_infectee_indices
            ):
                g.nodes[ni]["history"]["infected_at"] = self._cur_tick
                g.nodes[ni]["history"]["infected_by"] = ni_by

        cond = np.zeros_like(infectious_mask, dtype=np.bool)
        cond[newly_infected_indices] = True
        return cond

    def __will_be_hospitalized(
            self,
            data: DataDict,
            mask: np.ndarray) -> np.ndarray:

        num_new = mask.sum()
        if num_new == 0:
            return np.zeros(0)

        will_be_hospitalized_prob = self._infection.hospitalization_prob.rvs(
            num_new
        )

        # roll the dice
        will_be_hospitalized = (
            self._rstate.binomial(1, will_be_hospitalized_prob, size=num_new)
            == 1
        )

        return will_be_hospitalized

    def __will_die(
            self,
            data: DataDict,
            mask: np.ndarray) -> np.ndarray:

        num_new_hosp = mask.sum()
        will_die_prob = self._infection.death_prob.rvs(num_new_hosp)

        # roll the dice
        will_die = (
            self._rstate.binomial(1, will_die_prob, size=num_new_hosp) == 1
        )

        return will_die

    def __will_have_symptoms(
            self,
            data: DataDict,
            mask: np.ndarray) -> np.ndarray:
        """
        Calculate who will develop symptoms

        Returns:
            np.ndarray
                The array will be as long as the number of
                true entries in mask.
        """

        new_infec = mask
        num_new_infec = new_infec.sum()
        if num_new_infec == 0:
            return np.zeros(0)

        if isinstance(self._population, NetworkXPopulation):
            # read probs from graph
            symp_probs = nx.get_node_attributes(
                self._population._graph,
                "symp_prob"
            ).values()

            symp_prob = np.fromiter(
                symp_probs,
                count=len(symp_probs),
                dtype=np.float)[new_infec]
        else:
            symp_prob = self._infection.will_have_symptoms_prob.rvs(
                num_new_infec)

        will_have_symp = (
            self._rstate.binomial(1, symp_prob, size=num_new_infec) == 1
        )
        return will_have_symp

    def __reported_quarantine(self, data: DataDict) -> np.ndarray:
        if not self._measures.measures_active:
            return np.zeros(data.field_len, dtype=np.bool)

    def __quarantined_traced(self, data: DataDict) -> np.ndarray:

        if len(self._stat_collector["num_reported"]) <= self._cur_tick:
            self._stat_collector["num_reported"].append(0)
            self._stat_collector["num_traced"].append(0)
            self._stat_collector["num_traced_infected"].append(0)
            self._stat_collector["num_traced_infectee"].append(0)
            self._stat_collector["num_traced_infected_total"].append(0)

        if (not self._measures.contact_tracing or
                not self._measures.measures_active):
            return np.zeros(data.field_len, dtype=np.bool)

        if self._measures.population_tracking:
            # Only a subset of the population responds to measures
            base_mask = self.states["is_tracked"](data)
        else:
            base_mask = np.ones(data.field_len, dtype=np.bool)

        reported_mask = (
            self.states["is_reported"](data) &
            self.states["is_tracable"](data)
            )
        reported_mask = reported_mask & base_mask

        if reported_mask.sum() == 0:
            return np.zeros(data.field_len, dtype=np.bool)

        # Get contact history
        con_history = self.trace_contacts
        len_history = len(con_history)

        if len_history == 0:
            return np.zeros(data.field_len, dtype=np.bool)

        # Contacts from the same day are already in the history
        backtrack_length = self._measures.backtrack_length + 1

        if len_history < backtrack_length:
            backtrack_length = len_history

        """
        con_history = np.atleast_2d(
            np.squeeze(np.hstack(con_history[-backtrack_length:]))
        )
        """
        con_history = con_history[-backtrack_length:]
        tracked_reported_indices = np.nonzero(reported_mask)[0]

        self._stat_collector["num_reported"][-1] += reported_mask.sum()

        # Only check contacts of tracked infected people
        """
        contacted_indices = []
        for tracked_index in tracked_reported_indices:
            hist_ids = con_history[:, 0] == tracked_index
            contacts = con_history[hist_ids, 1]
            contacted_indices.append(contacts)
        contacted_indices = np.unique(np.concatenate(contacted_indices))
        """

        # TODO UPDATE TO DICT BASED
        def find_infectee_bt(infect_history, tracked_reported_indices):

            id_mask = np.bitwise_or.reduce(
                infect_history[:, 1] ==
                tracked_reported_indices[:, np.newaxis],
                axis=0)
            infectee_indices = np.unique(infect_history[id_mask, 0])

            return infectee_indices

        def find_ci(con_history, tracked_reported_indices):
            contacts = []
            for day in con_history:

                for ti in tracked_reported_indices:
                    this_contacts = day[ti]
                    contacts += this_contacts

            return np.unique(contacts)

        contacted_indices = find_ci(con_history, tracked_reported_indices)

        """
        print(
            reported_mask.sum(),
            data["is_quarantined"].sum(),
            len(contacted_indices))
        """
        if len(contacted_indices) == 0:
            return np.zeros(data.field_len, dtype=np.bool)

        contacted_mask = np.zeros(data.field_len, dtype=np.bool)
        contacted_mask[contacted_indices] = True

        self._stat_collector["num_traced"][-1] += contacted_mask.sum()
        self._stat_collector["num_traced_infected"][-1] += (
            data["is_infected"][contacted_mask].sum())

        # backwards traces

        infect_history = np.atleast_2d(np.squeeze(
            np.hstack(self.trace_infection[-backtrack_length:])))

        infectee_ids = find_infectee_bt(
            infect_history, tracked_reported_indices)

        self._stat_collector["num_traced_infectee"][-1] += len(infectee_ids)

        if len(infectee_ids) > 0:
            contacted_mask[infectee_ids] = True

        if not self._measures.track_uninfected:
            infected_mask = self.states["is_infected"](data)
            contacted_mask[~infected_mask] = False

        contacted_mask &= base_mask

        # apply tracing efficiency

        is_suc_traced = self._rstate.binomial(
            1,
            self._measures.tracing_efficiency,
            size=data.field_len) == 1
        contacted_mask &= is_suc_traced

        if isinstance(self._population, NetworkXPopulation):
            contacted_indices = np.nonzero(contacted_mask)[0]
            # update graph history
            g = self._population._graph
            for ci in contacted_indices:
                if "traced" not in g.nodes[ci]["history"]:
                    g.nodes[ci]["history"]["traced"] = []
                g.nodes[ci]["history"]["traced"].append(self._cur_tick)

        self._stat_collector["num_traced_infected_total"][-1] += (
            data["is_infected"][contacted_mask].sum())
        return contacted_mask

        """
        if self._measures.is_SOT_active:

            is_contactable_indices = np.nonzero(tracked_mask)[0]

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
                        if not self._measures.track_uninfected:
                            SOT_contacts_mask[~infected_mask] = False

            contacted_mask = np.logical_or(contacted_mask, SOT_contacts_mask)

        cond = np.logical_or(tracked_infected_mask, contacted_mask)
        """

    def __will_be_tested(self, data: DataDict) -> np.ndarray:
        if not self._measures.measures_active:
            return np.zeros(data.field_len, dtype=np.bool)

        if not self._measures.testing:
            return np.zeros(data.field_len, dtype=np.bool)

        # Testing for newly quarantined or negative result after first test

        is_quarantined = self.states["is_quarantined"](data)
        new_quarantined = self.states["is_new_quarantined"](data)
        needs_second_test = (
            is_quarantined &
            ~self.states["time_until_second_test_result"](data)
        )

        cond = new_quarantined | needs_second_test
        return cond

    def __will_test_negative(
            self,
            data: DataDict,
            mask: np.ndarray) -> np.ndarray:

        num_mask = mask.sum()
        is_infectious_mask = self.states["is_infectious"](data)[mask]
        is_symptomatic_mask = self.states["is_symptomatic"](data)[mask]
        time_since_infect = data["time_since_infectious"][mask]

        always_negative_mask = ~is_infectious_mask

        always_negative_num = np.sum(always_negative_mask)
        always_negative_indices = np.nonzero(always_negative_mask)[0]
        always_negative = self._rstate.binomial(
            1,
            self._measures.test_false_positive_pdf.rvs(always_negative_num)
            ) == 0

        # default is will_test_negative = True
        result = np.ones(num_mask, dtype=np.bool)
        result[always_negative_indices] = always_negative
        # Assume that symptomatic always get tested positive

        result[is_symptomatic_mask] = False

        infectious_to_test_mask = is_infectious_mask & (~is_symptomatic_mask)
        infectious_to_test_indices = np.nonzero(infectious_to_test_mask)[0]

        if not np.any(infectious_to_test_mask):
            return result

        infectious_dur = time_since_infect[infectious_to_test_mask]

        correct_test_prob = self._measures.test_efficiency(infectious_dur)
        tests_positive = self._rstate.binomial(1, correct_test_prob) == 1

        result[infectious_to_test_indices] = ~tests_positive

        return result

    def __quarantine_test_free(
            self,
            data: DataDict):

        mask = (
            data["time_since_last_test_result"] >
            self._measures.quarantine_duration.rvs(data.field_len)
            )

        return mask

    def __tracing_active(self, data: DataDict) -> np.ndarray:
        if self._measures.measures_active:
            return np.ones(data.field_len, dtype=np.bool)
        else:
            return np.zeros(data.field_len, dtype=np.bool)

    def __takes_random_test(self, data: DataDict) -> np.ndarray:

        eligible = (
            ~self._states["is_quarantined"](data) &
            ~(
                self._states["is_recovered"](data) &
                self._states["is_index_case"](data)
            ) &
            ~self._states["is_symptomatic"](data)
        )

        eligible_indices = np.nonzero(eligible)[0]

        if isinstance(self._population, NetworkXPopulation):

            g = self._population._graph

            for ei in eligible_indices:
                if not g.nodes[ei]["random_testable"]:
                    eligible[ei] = False
        eligible_indices = np.nonzero(eligible)[0]
        num_eligible = np.sum(eligible)

        if (
                isinstance(self._population, NetworkXPopulation) and
                (self._measures.random_test_mode == "lin weight")
           ):
            g = self._population._graph

            weights = np.asarray(list(dict(g.degree)), dtype=np.float)[eligible_indices]
            weights /= weights.sum()
            nzero = np.sum(weights > 0)
            num_tests = min(self._measures.random_test_num, nzero)
            if num_tests == nzero:
                takes_random_test = np.zeros(
                    data.field_len,
                    dtype=np.bool)
                takes_random_test[eligible_indices] = True
            else:
                test_indices = self._rstate.choice(
                    eligible_indices,
                    size=num_tests,
                    p=weights,
                    replace=False)
                takes_random_test = np.zeros(
                    data.field_len,
                    dtype=np.bool)
                takes_random_test[test_indices] = True

            """
            sorted_indices = sorted(
                eligible_indices,
                key=lambda i: g.degree[i],
                reverse=True)
            """
        elif (
                isinstance(self._population, NetworkXPopulation) and
                (self._measures.random_test_mode == "distribute class")
             ):
            g = self._population._graph

            clique_size = config["population"]["nx"]["kwargs"]["clique_size"]
            n_cliques = g.graph["n_school"] // clique_size

            n_tests_per_class = self._measures.random_test_num / n_cliques
            min_tests_per_class = int(np.floor(n_tests_per_class))
            remaining_tests = int(
                self._measures.random_test_num - min_tests_per_class*n_cliques)

            tests_per_class = (
                np.zeros(n_cliques, dtype=np.int) + min_tests_per_class
            )
            rnd_classes = self._rstate.choice(
                np.arange(n_cliques),
                size=remaining_tests,
                replace=False)
            tests_per_class[rnd_classes] += 1
            takes_random_test = np.zeros(data.field_len, dtype=np.bool)
            class_indices = np.arange(clique_size)
            for i, tests in enumerate(tests_per_class):
                test_indices = self._rstate.choice(
                    class_indices,
                    size=tests,
                    replace=False) + i*clique_size

                takes_random_test[test_indices] = eligible[test_indices]

        else:
            if self._measures.random_test_num >= num_eligible:
                return eligible
            else:
                takes_random_test = np.zeros(data.field_len, dtype=np.bool)
                rnd_indices = self._rstate.choice(
                    eligible_indices,
                    size=self._measures.random_test_num,
                    replace=False)
                takes_random_test[rnd_indices] = True
        return takes_random_test

    def __correlated_latent(self, data, rows):
        incub_time = data["time_until_symptoms"][rows]

        num_rows = rows.sum()
        latent_duration = self._infection.latent_duration.rvs(num_rows)

        latent_duration = (incub_time - latent_duration).astype(np.int)
        latent_duration[latent_duration < 0] = 0

        return latent_duration
