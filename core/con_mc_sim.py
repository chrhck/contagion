"""
Name: fd_roll_dice_social.py
Authors: Stephan Meighen-Berger, Martina Karl
Runs a monte-carlo (random walk) simulation
for the social interactions.
"""

"Imports"
from sys import exit
import numpy as np
from time import time
from numpy.random import choice


class CON_mc_sim(object):
    """
    class: CON_mc_sim
    Monte-carlo simulation for the infection spread.
    Parameters:
        -int infected:
            The starting infected population
        -np.array population:
            The population
        -obj infection:
            The infection object
        -np.array tracked:
            The tracked population
        -obj log:
            The logger
        -dic config:
            Dictionary from the config file
    Returns:
        -None
    "God does not roll dice!"
    """

    def __init__(self, infected, population, infection, tracked, log, config):
        """
        function: __init__
        Initializes the class.
        Parameters:
            -int infected:
                The starting infected population
            -np.array population:
                The population
            -obj infection:
                The infection object
            -np.array tracked:
                The tracked population
            -obj log:
                The logger
            -dic config:
                Dictionary from the config file
        Returns:
            -None
        """
        self.__log = log
        self.__config = config
        self.__infect = infection
        self.__dt = config["time step"]
        self.__t = np.arange(0.0, self.__config["simulation length"], step=self.__dt)
        self.__log.debug("The interaction intensity pdf")
        if self.__config["interaction intensity"] == "uniform":
            self.__intense_pdf = self.__intens_pdf_uniform
            # The Reproductive Number
            self.__R = (
                self.__config["mean social circle interactions"]
                * self.__config["infection duration mean"]
                * 0.5
            )
        else:
            self.__log.error(
                "Unrecognized intensity pdf! Set to "
                + self.__config["interaction intensity"]
            )
            exit("Check the interaction intensity in the config file!")
        self.__log.debug("Constructing simulation population")
        self.__log.debug("The infected ids and durations...")
        infect_id = np.random.choice(
            range(len(population)), size=infected, replace=False
        )
        infect_dur = np.around(self.__infect.pdf_duration(infected).flatten())
        # Constructing population array
        # Every individual has 5 components
        #   -individual's id
        #   -ids of individuals in sc
        #   -The number of interactions per time step
        #   -infected
        #   -remaining duration of infection
        #   -immune
        self.__log.debug("Filling the population array")
        self.__population = np.array(
            [
                [
                    i,  # The person's id
                    population[i][0],  # The social circle
                    population[i][1],  # The encouter rate
                    False,  # Infected?
                    0.0,  # Infection duration
                    False,  # Immunity?
                ]
                for i in range(len(population))
            ]
        )
        # Adding the infected
        for i, id_inf in enumerate(infect_id):
            self.__population[id_inf][3] = True
            self.__population[id_inf][4] = infect_dur[i]
        self.__log.info("There will be %d simulation steps" % len(self.__t))
        # Removing social mobility of tracked people
        if tracked is not None:
            for i in tracked:
                self.__population[i][2] = 0
        if self.__config["save population"]:
            self.__log.debug("Saving the distribution of infected")
            self.__distribution = []
            self.__total_infections = []
            self.__new_infections = []
            self.__immune = []
        # Running the simulation
        start = time()
        self.__simulation()
        end = time()
        self.__log.info("MC simulation took %f seconds" % (end - start))

    @property
    def population(self):
        """
        function: population
        Returns the population
        Parameters:
            -None
        Returns:
            -np.array population:
                The current population
        """
        return self.__population

    @property
    def distribution(self):
        """
        function: distribution
        Returns the distribution
        Parameters:
            -None
        Returns:
            -np.array distribution:
                The distribution
        """
        return self.__distribution

    @property
    def infections(self):
        """
        function: infections
        Returns the infections
        Parameters:
            -None
        Returns:
            -np.array infections:
                The total infections
        """
        return np.array(self.__total_infections)

    @property
    def new_infections(self):
        """
        function: new_infections
        Returns the new_infections
        Parameters:
            -None
        Returns:
            -np.array new_infections:
                The total new_infections
        """
        return np.array(self.__new_infections)

    @property
    def immune(self):
        """
        function: immune
        Returns the immune
        Parameters:
            -None
        Returns:
            -np.array immune:
                The total immune
        """
        return np.array(self.__immune)

    @property
    def time_array(self):
        """
        function: time_array
        Returns the time array used
        Parameters:
            -None
        Returns:
            np.array __t:
                The time array
        """
        return self.__t

    @property
    def R(self):
        """
        function: reproductive number
        Average number of infections due to
        one patient (not assuming measures were taken)
        Parameters:
            -None
        Returns:
            -float R:
                The reproductive number
        """
        return self.__R

    def __simulation(self):
        """
        function: __simulation
        Runs the simulation
        Parameters:
            -None
        Returns:
            -None
        """
        self.__infections = []
        for _ in self.__t:
            # TODO: This needs to be optimized to a comprehension
            new_infections = []
            for person_id, person in enumerate(self.__population):
                # If infected
                if person[3]:
                    # Infected people contacts others
                    contacts = choice(person[1], size=person[2])
                    # Intensity of the contacts
                    intens_arr = self.__intense_pdf(len(contacts)).flatten()
                    infection_prob = np.array(
                        [self.__infect.pdf(intens) for intens in intens_arr]
                    )
                    # Are they infected?
                    infection = np.around(infection_prob)
                    new_infections.append(
                        [
                            contacts[i]
                            for i in range(len(infection))
                            if infection[i] > 0.0
                        ]
                    )
                    # subtracting 1 from the infection duration
                    if self.__population[person_id][4] > 0:
                        self.__population[person_id][4] -= 1
                        # Person has survived infection
                        if self.__population[person_id][4] == 0:
                            self.__population[person_id][3] = False
                            self.__population[person_id][5] = True
                # If Susceptible
                elif not person[5]:
                    # Susceptible person others
                    contacts = choice(person[1], size=person[2])
                    # Are there any infected?
                    n_inf_cont = np.sum(
                        [
                            person[3]
                            for person in self.__population
                            if (person[0] in contacts)
                        ]
                    )
                    if n_inf_cont > 0:
                        # Intensity of the contacts
                        intens_arr = self.__intense_pdf(n_inf_cont).flatten()
                        infection_prob = np.array(
                            [self.__infect.pdf(intens) for intens in intens_arr]
                        )
                        # Are they infected?
                        infection = np.around(infection_prob)
                        if np.any(infection):
                            new_infections.append([person_id])
            # Adding newly infected to the population
            new_infection_counter = 0
            for new_infect in new_infections:
                # The durations
                tmp_dur = np.around(
                    self.__infect.pdf_duration(len(new_infect)).flatten()
                )
                for i, id_infect in enumerate(new_infect):
                    if (not self.__population[id_infect][3]) & (
                        not self.__population[id_infect][5]
                    ):
                        self.__population[id_infect][3] = True
                        self.__population[id_infect][4] = tmp_dur[i]
                        new_infection_counter += 1
            if self.__config["save population"]:
                self.__distribution.append(np.copy(self.__population))
                self.__total_infections.append(len(self.__fetch_infected()))
                self.__new_infections.append(new_infection_counter)
                self.__immune.append(len(self.__fetch_immune()))

    def __intens_pdf_uniform(self, contacts):
        """
        function: __intens_pdf_uniform
        The social interaction intensity
        drawn from a uniform distribution
        Parameters:
            -int contacts:
                Number of contacts
        Returns:
            -np.array res:
                The contact intensities
        """
        return np.random.uniform(low=0.0, high=1.0, size=contacts)

    def __fetch_infected(self):
        """
        function: __fetch_infected
        Helper function to fetch the infected count
        Parameters:
            -None
        Returns:
            -np.array res:
                The number of currently infected
        """
        return np.array([person for person in self.__population if person[3]])

    def __fetch_immune(self):
        """
        function: __fetch_immune
        Helper function to fetch the immune count
        Parameters:
            -None
        Returns:
            -np.array res:
                The number of currently infected
        """
        return np.array([person for person in self.__population if person[5]])
