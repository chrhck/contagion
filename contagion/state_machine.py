import abc
from collections import defaultdict
from typing import Dict, Any, Callable, Union, List

import numpy as np
import pandas as pd
from scipy import sparse

from .infection import Infection
from .pdfs import PDF


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
    def from_boolean(cls, name: str):
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

    def __init__(self, name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = name

    @abc.abstractmethod
    def __call__(self, df):
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

    def __call__(self, df):
        """
        Perform the transition.

        All rows in `df` which where previusly in state A are transitioned
        to state B.

        Parameters:
            df: pd.DataFrame
        """

        # Invert state B to select all rows which are _not_ in state B
        # Use state A as condition so that only rows are activated which
        # where in state A
        (~self._state_b).change_state(df, True, self._state_a(df))

        self._state_a.change_state(df, False)


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

    def __call__(self, df: pd.DataFrame):
        """Evaluate condition on DataFrame"""
        return self.condition(df)


class ConditionalTransitionMixin(object):

    def __init__(self, condition, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._condition = condition


class ConditionalTransition(_Transition, ConditionalTransitionMixin):
    def __init__(
            self,
            name: str,
            state_a: _State,
            state_b: _State,
            condition: Union[_State, np.ndarray, pd.DataFrame],
            *args, **kwargs,
           ):

        _Transition.__init__(self, name, *args, **kwargs)
        ConditionalTransitionMixin.__init__(self, condition, *args, **kwargs)
        self._state_a = state_a
        self._state_b = state_b

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
            state_a: FloatState,
            *args, **kwargs
           ):
        super().__init__(name, *args, **kwargs)
        self._state_a = state_a

    def __call__(
            self,
            df: pd.DataFrame):

        # Get state
        cur_state = self._state_a.get_state_value(df)
        self._state_a.change_state(df, cur_state-1)


class InitializeTimerTransition(_Transition, ConditionalTransitionMixin):
    def __init__(
            self,
            name: str,
            state_a: FloatState,
            initialization_pdf: PDF,
            condition: Union[_State, np.ndarray, pd.DataFrame],
            *args,
            **kwargs
           ):

        _Transition.__init__(self, name, *args, **kwargs)
        ConditionalTransitionMixin.__init__(self, condition)
        self._state_a = state_a
        self._initialization_pdf = initialization_pdf

    def __call__(
            self,
            df: pd.DataFrame):

        # Invert state to get all rows that are currently 0
        zero_rows = ~(self._state_a)

        num_zero_rows = zero_rows.sum(axis=0)

        initial_vals = self._initialization_pdf.rvs(num_zero_rows)
        zero_rows.change_state(df, initial_vals)

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


class StateMachine(object):

    def __init__(
            self,
            states: Dict,
            transitions: Dict,
            df: pd.DataFrame,
            *args, **kwargs):
        self._transitions = transitions
        self._states = states
        self._df = df

    @property
    def states(self):
        return self._states

    @property
    def transitions(self):
        return self._transitions


class ContagionStateMachine(StateMachine):

    def __init__(
            self,
            states: Dict,
            transitions: Dict,
            df: pd.DataFrame,
            interactions: sparse.spmatrix,
            infection: Infection,
            rstate: np.random.RandomState,
            *args, **kwargs):
        super().__init__(states, transitions, df, *args, **kwargs)

        self._interactions = interactions
        self._rstate = rstate
        self._infection = infection
        self._statistics = defaultdict(list)

    def __get_new_infections(self) -> np.ndarray:
        pop_csr = self._interactions.tocsr()

        # TODO: use state
        infected_mask = self.states["is_infected"](self._df)
        infected_indices = infected_mask.index[infected_mask]

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
        contact_strength = self.__intense_pdf(num_succesful_contacts)
        infection_prob = self.__infect.pdf_infection_prob(contact_strength)

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

        cond = np.zeros(len(df), dtype=np.bool)
        cond[newly_infected_indices] = True

        return cond
