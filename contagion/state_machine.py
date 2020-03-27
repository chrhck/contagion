import abc
from typing import Dict, Any, Callable, Union, List

import numpy as np
import pandas as pd


class _State(object, metaclass=abc.ABCMeta):
    """Interface for all States"""

    def __init__(
            self,
            state_getter: Callable,
            state_value_getter: Callable,
            name: str,
            state_change: Callable):
        self._state_getter = state_getter
        self._state_value_getter = state_value_getter
        self._name = name
        self._state_change = state_change

    def __call__(self, df: pd.DataFrame):
        """
        Returns the state

        Parameters:
            df: pd.DataFrame

        Returns:
            pd.Series
        """
        return self._state_getter(df)

    def get_state_value(self, df: pd.DataFrame):
        return self._state_value_getter(df)

    def __invert__(self):
        """
        Return a state with inverted condition
        """
        def inverted_condition(df):
            return ~(self(df))
        return type(self)(
            inverted_condition,
            self._state_value_getter,
            "inverted_" + self.name,
            self._state_change)

    @abc.abstractmethod
    def change_state(
            self,
            df: pd.DataFrame,
            state: np.ndarray,
            condition=None):
        pass

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
    def from_simple_boolean(cls, name: str):
        def get_state(df):
            return df[name]

        def state_change(df: pd.DataFrame, state: bool, condition):
            df.loc[condition, name] = state
        return cls(get_state, get_state, name, state_change)

    def change_state(
            self,
            df: pd.DataFrame,
            state: np.ndarray,
            condition=None,
            ):
        """Changes the state in the DataFrame"""
        # Check which is currently in this state

        cond = pd.Series(self(df), copy=True)

        if isinstance(condition, _State):
            cond &= condition(df)
        elif isinstance(condition, np.ndarray):
            cond &= condition
        elif isinstance(condition, pd.Series):
            cond &= condition
        elif condition is None:
            pass
        else:
            raise ValueError("Unsupported type: ", type(condition))
        self._state_change(df, state, cond)


class FloatState(_State):

    @classmethod
    def from_timer(cls, name: str):
        def get_state(df):
            return df[name] > 0

        def get_state_value(df):
            return df[name]

        def state_change(df: pd.DataFrame, state: np.ndarray, condition):
            df.loc[condition, name] = state

        return cls(get_state, get_state_value, name, state_change)

    def change_state(
            self,
            df: pd.DataFrame,
            state: np.ndarray,
            condition=None):
        """Changes the state in the DataFrame"""
        # Check which is currently in this state

        cond = self(df)

        if isinstance(condition, _State):
            cond &= condition(df)
        elif isinstance(condition, np.ndarray):
            cond &= condition
        elif condition is None:
            pass
        else:
            raise ValueError("Unsupported type: ", type(condition))

        self._state_change(df, state, cond)


class _Transition(object, metaclass=abc.ABCMeta):
    """Interface for Transitions"""

    @abc.abstractmethod
    def __call__(self, df):
        pass


class Transition(_Transition):

    def __init__(
            self,
            name: str,
            state_a: _State,
            state_b: _State):

        self._name = name
        self._state_a = state_a
        self._state_b = state_b

    def __call__(self, df):
        """
        Perform the transition

        Parameters:
            df: pd.DataFrame
        """

        self._state_a.change_state(df, False)
        self._state_b.change_state(df, True)


class Condition(object):
    """
    Convenience class for storing references to conditions
    """
    def __init__(self, condition):
        self._condition = condition

    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, val):
        self._condition = val


class ConditionalTransition(_Transition):
    def __init__(
            self,
            name: str,
            state_a: _State,
            state_b: _State,
            condition: Union[_State, np.ndarray, pd.DataFrame]
           ):

        self._name = name
        self._state_a = state_a
        self._state_b = state_b
        self._condition = condition

    def __call__(
            self,
            df: pd.DataFrame):

        if isinstance(self._condition, _State):
            cond = self._condition(df)
        elif isinstance(self._condition, np.ndarray):
            cond = self._condition
        elif isinstance(self._condition, pd.Series):
            cond = self._condition
        elif self._condition is None:
            pass
        else:
            raise ValueError("Unsupported type: ", type(self._condition))

        (~self._state_b).change_state(df, True, cond & self._state_a(df))
        self._state_a.change_state(df, False, self._condition)


class DecreaseTimerTransition(_Transition):
    def __init__(
            self,
            name: str,
            state_a: _State,
           ):

        self._name = name
        self._state_a = state_a

    def __call__(
            self,
            df: pd.DataFrame):

        # Get state
        cur_state = self._state_a.get_state_value(df)
        self._state_a.change_state(df, cur_state-1)


def MultiStateTransition(object):
    def __init__(
            self,
            name: str,
            state_a: _State,
            states_b: List[_State]):

        self._name = name
        self._state_a = state_a
        self._states_b = states_b

    def __call__(self, df):
        """
        Perform the transition

        Parameters:
            df: pd.DataFrame
        """

        self._state_a.change_state(df)

        for state in self._states_b:
            state.change_state(df)


def MultiStateConditionalTransition(object):
    def __init__(
            self,
            name: str,
            state_a: _State,
            states_b: List[_State]):

        self._name = name
        self._state_a = state_a
        self._states_b = states_b

    def __call__(
            self,
            df: pd.DataFrame,
            condition: Union[_State, np.ndarray]):
        """
        Perform the transition

        Parameters:
            df: pd.DataFrame
        """

        self._state_a.change_state(df.loc[condition])

        for state in self._states_b:
            state.change_state(df.loc[condition])


simple_boolean_state_names = [
    "is_infected", "has_died", "is_removed", "is_infectious",
    "is_hospitalized", "is_recovering", "in_incubation", "has_recovered"]

simple_boolean_states = {name: BooleanState.from_simple_boolean(name)
                         for name in simple_boolean_state_names}


timer_state_names = [
    "incubation_duration", "hospitalization_duration", "recovery_time",
    "time_until_hospitalization"]


timer_states = {name: FloatState.from_timer(name)
                for name in simple_boolean_state_names}


"""
transitions = [
    ConditionalTransition(
        "healthy_incubation",
        ~simple_boolean_states["is_infected"],
        simple_boolean_states["in_incubation"],
        Condition(True)
    )
    ConditionalTransition(
        "incubation_infectious",
        simple_boolean_states["in_incubation"],
        simple_boolean_states["is_infectious"],
        timer_states["incubation_duration"]


    )
    ]
"""

"""
class StateMachine(object):

    def __init__(self, states, transitions: Dict):
        self._transitions = transitions
"""
