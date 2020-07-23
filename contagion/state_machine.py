# -*- coding: utf-8 -*-

"""
Name: state_machine.py
Authors: Christian Haack, Stephan Meighen-Berger, Andrea Turcati

Base implementation of states, conditions, transitions and the state machine.
"""

from __future__ import annotations
import abc
from collections import defaultdict
from copy import deepcopy
import functools
import logging
from typing import Any, Callable, Union, List, Tuple, Dict, Optional

import numpy as np  # type: ignore
import networkx as nx  # type: ignore
import pandas as pd  # type: ignore

from .config import config

from .pdfs import PDF

_log = logging.getLogger(__name__)

DEBUG = False
if DEBUG:
    _log.warn("DEBUG flag enabled. This will drastically slow down the code")


class DataDict(Dict[str, np.ndarray]):
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
        state_value_getter: Callable[[np.ndarray], np.ndarray]
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
            TCondition as third argument.

    """

    def __init__(
        self,
        state_getter: Callable[[np.ndarray], np.ndarray],
        state_value_getter: Callable[[np.ndarray], np.ndarray],
        name: str,
        data_field: str,
        state_change: Callable[[DataDict, np.ndarray, TCondition], None],
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
        in a DataDict. This state is active when the state value is > 0.
        The name of the state corresponds to the data field name
        in the DataDict.

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
        in a DataDict. This state is active when the state value is > -inf.
        The name of the state corresponds to the data field
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
        pipe_condition_mask: bool
            If true, the call to this transition returns the condition_mask
    """

    def __init__(
            self,
            name: str,
            pipe_condition_mask: bool = False,
            *args,
            **kwargs):
        self._name = name
        self._pipe_condition_mask = pipe_condition_mask

    @abc.abstractmethod
    def __call__(
            self,
            data: DataDict,
            condition_mask: Optional[np.ndarray] = None) -> np.ndarray:
        pass

    @property
    def name(self) -> str:
        return self._name


class Transition(_Transition):
    """
    Basic Transition class

    Transitions all rows which are in `state_a` to `state_b`. This
    sets `state_a`  to False and `state_b` to True.

    Parameters:
        name: str
        state_a: _State
        state_b: _State

    """
    def __init__(
            self,
            name: str,
            state_a: _State,
            state_b: _State,
            pipe_condition_mask: bool = False,
            *args, **kwargs
    ):

        super().__init__(name, pipe_condition_mask, *args, **kwargs)
        self._state_a = state_a
        self._state_b = state_b

    @log_call
    def __call__(
            self,
            data: DataDict,
            condition_mask: Optional[np.ndarray] = None) -> np.ndarray:
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

        if self._pipe_condition_mask:
            return condition_mask
        return changed


class ChangeStateConditionalTransition(_Transition, ConditionalMixin):
    """
    Change a state where external condition is true

    Parameters:
        name: str
        state_a: Union[_State, Tuple[_State, Union[bool,
                Callable[[DataDict, TCondition], np.ndarray]]]
            State to change. Can either be a `_State`, in which case the state
            will be set to true or a tuple, where the second item
            is the value the state should be set to or a Callable returning the
            value.
        condition: TCondition
        pipe_condition_mask: bool
            If true, the call to this transition will return a boolean
            `np.ndarray` encoding the rows for which `condition` is True.

    """

    _state_a: _State
    _state_a_val: bool

    def __init__(
        self,
        name: str,
        state_a: Union[
            _State,
            Tuple[
                _State,
                Union[bool, Callable[[DataDict, TCondition], np.ndarray]
            ]
        ],
        condition: TCondition = None,
        pipe_condition_mask: bool = False,
        *args,
        **kwargs,
    ):
        _Transition.__init__(self, name, pipe_condition_mask, *args, **kwargs)
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
            data: DataDict,
            condition_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Perform the transition.

        All rows in data which are in state A and for which the
        external condition is true are transitioned to a new value.

        Parameters:
            data: DataDict
            condition_mask: Optional[np.ndarray]
                Additional condition in form of a boolean np.ndarray.
        """
        cond = self.unify_condition(data)

        if condition_mask is not None:
            cond = cond & condition_mask

        if callable(self._state_a_val):
            parsed_val = self._state_a_val(data, cond & self._state_a(data))
        else:
            parsed_val = self._state_a_val

        changed = self._state_a.change_state(data, parsed_val, cond)

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
        condition: TCondition,
        pipe_condition_mask: bool
            If true, the call to this transition will return a boolean
            `np.ndarray` encoding the rows for which `condition` is True.
    """

    def __init__(
        self,
        name: str,
        state_a: _State,
        state_b: _State,
        condition: TCondition,
        pipe_condition_mask: bool = False,
        *args,
        **kwargs,
    ):

        _Transition.__init__(self, name, pipe_condition_mask, *args, **kwargs)
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

        if self._pipe_condition_mask:
            return condition_mask
        return changed


class DecreaseTimerTransition(_Transition, ConditionalMixin):
    """
    Decrease the value of a FloatState by one

    Parameters:
        name: str
        state_a: FloatState
        condition: Optional[TCondition],
        pipe_condition_mask: bool
            If true, the call to this transition will return a boolean
            `np.ndarray` encoding the rows for which `condition` is True.
    """

    def __init__(
        self,
        name: str,
        state_a: FloatState,
        condition: TCondition = None,
        pipe_condition_mask: bool = False,
        *args,
        **kwargs,
    ):
        _Transition.__init__(self, name, pipe_condition_mask, *args, **kwargs)
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

        if self._pipe_condition_mask:
            return condition_mask
        return changed


class IncreaseTimerTransition(_Transition, ConditionalMixin):
    """
    Increase the value of a FloatState by one

    Parameters:
        name: str
        state_a: FloatState
        condition: Optional[TCondition],
        pipe_condition_mask: bool
            If true, the call to this transition will return a boolean
            `np.ndarray` encoding the rows for which `condition` is True.
    """

    def __init__(
        self,
        name: str,
        state_a: FloatState,
        condition: TCondition = None,
        pipe_condition_mask = False,
        *args,
        **kwargs,
    ):
        _Transition.__init__(self, name, pipe_condition_mask, *args, **kwargs)
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

        if self._pipe_condition_mask:
            return condition_mask
        return changed


class InitializeTimerTransition(_Transition, ConditionalMixin):
    """
    Initialize a timer state to values drawn from a PDF

    Parameters:
        name: str
        state_a: FloatState
        initialization_pdf: Optional[PDF]
            This pdf is used to initialize the state values upon activation.
        condition: Optional[TCondition]
        pipe_condition_mask: bool
            If true, the call to this transition will return a boolean
            `np.ndarray` encoding the rows for which `condition` is True.
        stateful_init_func: Optional[
                Callable[[DataDict, np.ndarray], np.ndarray]]
            Supply a Callable that takes a DataDict as first and a boolean mask
            encoding the inactive states as second argument when
            `initialization_pdf` is None. The callable will be used to
            initialize the state.
    """

    def __init__(
        self,
        name: str,
        state_a: FloatState,
        initialization_pdf: Optional[PDF] = None,
        condition: TCondition = None,
        pipe_condition_mask: bool = False,
        stateful_init_func: Callable[[DataDict, np.ndarray], np.ndarray]=None,
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
                self._statistics[
                    field + "_" + field2 + "_" + str(state)].append(cond.sum())

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

    The state machine operates on a dictionary of numpy arrays (`DataDict`).
    Each key in the dictionary represents a state, which can be different for
    each row in the numpy array.
    The state machine holds a list of transitions which are applied in sequence
    to the data.

    Subclasses have to implement the `transitions` and `state` properties.

    Parameters:
        data: Union[pd.DataFrame, DataDict]
            Can be either a DataFrame of `DataDict`
        stat_collection: Optional[StatCollector]
            Stat collector object
        trace_states: bool
            Record the current state of `data` after each tick.
            Warning: Memory intensive

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
