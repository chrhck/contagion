from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import networkx as nx

from .config import config
from .infection import Infection
from .measures import Measures
from .population import Population, NetworkXPopulation
from .state_machine import (
    BooleanState, ChangeStateConditionalTransition, Condition,
    ConditionalTransition, DecreaseTimerTransition, DataDict,
    FloatState, IncreaseTimerTransition, InitializeCounterTransition,
    InitializeTimerTransition,
    MultiStateConditionalTransition, StateMachine, StatCollector,
    TransitionChain, _State)


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

        self._total_tests_today = 0

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
            boolean_state_names += ["is_tracable", "is_tracked"]
        if self._measures.quarantine:
            boolean_state_names += ["is_quarantined", "is_reported"]
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
            timer_state_names += [
                "time_until_test", "time_until_test_result",
                "time_until_second_test", "time_until_second_test_result",
            ]
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
                        pipe_condition_mask=True
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
                        ),
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
                                "symptomatic_quarantined",
                                (
                                    ~boolean_states["is_quarantined"],
                                    self._check_test_capacity
                                )
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
                                        (
                                            ~boolean_states["is_quarantined"],
                                            self._check_test_capacity
                                        )
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

        self._total_tests_today = 0

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
        if isinstance(self._population, NetworkXPopulation):
            symp_indices = np.nonzero(new_infec)[0][will_have_symp]
            # update graph history
            g = self._population._graph
            for si in symp_indices:
                g.nodes[si]["history"]["symptomatic"] = self._cur_tick
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

    def _check_test_capacity(
            self,
            data: DataDict,
            mask: np.ndarray) -> np.ndarray:
        if self._measures.test_capacity is not None:
            test_cap_today = self._measures.test_capacity(self._cur_tick)
            new_tests = mask.sum()
            if new_tests + self._total_tests_today > test_cap_today:
                rnd_ind = self._state.choice(
                    np.nonzero(mask)[0],
                    size=test_cap_today - self._total_tests_today)
                mask = np.zeros_like(mask, dtype=np.bool)
                mask[rnd_ind] = True
        self._total_tests_today += mask.sum()
        return mask

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
