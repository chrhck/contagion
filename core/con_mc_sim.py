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

    def __init__(
            self,
            infected,
            population,
            infection,
            tracked,
            log,
            config,
            rstate = None):
        """
        function: __init__
        Initializes the class.
        Parameters:
            -int infected:
                The starting infected population
            -np.array population:
                The population matrix
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
        self.__dt = config['time step']
        self.__pop_matrix = population
        self.__t = np.arange(
            0., self.__config['simulation length'],
            step=self.__dt
        )
        self.__log.debug('The interaction intensity pdf')
        if self.__config['interaction intensity'] == 'uniform':
            self.__intense_pdf = self.__intens_pdf_uniform
            # The Reproductive Number
            self.__R = (
                self.__config['mean social circle interactions'] *
                self.__config['infection duration mean'] * 0.5
            )
        else:
            self.__log.error('Unrecognized intensity pdf! Set to ' +
                             self.__config['interaction intensity'])
            exit('Check the interaction intensity in the config file!')
       
        if rstate is None:          
            self.__log.warning("No random state given, constructing new state")
            rstate = np.random.RandomState()
        self.__rstate = rstate

        self.__log.debug('Constructing simulation population')
        self.__log.debug('The infected ids and durations...')
        
        pop_size = len(population)

        infect_id = self.__rstate.choice(
            range(pop_size),
            size=infected,
            replace=False)
        infect_dur = np.around(
            self.__infect.pdf_duration(infected))
        # Constructing population array
        # Every individual has 5 components
        #   -individual's id
        #   -infected
        #   -remaining duration of infection
        #   -immune
        self.__log.debug('Filling the population array')
        
        self.__population = np.empty((pop_size, 4))
        self.__population[:, 0] = np.arange(pop_size)
        self.__population[:, 1] = 0
        self.__population[:, 2] = 0
        self.__population[:, 3] = 0
        
         # Adding the infected
        self.__population[infect_id, 1] = 1
        self.__population[infect_id, 2] = infect_dur

       
        self.__log.info('There will be %d simulation steps' %len(self.__t))
        # Removing social mobility of tracked people
        if tracked is not None:
           # TODO make this configurable

           self.__pop_matrix[tracked] = 0


        if self.__config['save population']:
            self.__log.debug("Saving the distribution of infected")
            self.__distribution = []
            self.__total_infections = []
            self.__new_infections = []
            self.__immune = []
        # Running the simulation
        start = time()
        self.__simulation()
        end = time()
        self.__log.info('MC simulation took %f seconds' % (end-start))

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

        population_size = len(self.__population)
        for _ in self.__t:
            #TODO: This needs to be optimized to a comprehension
            new_infections = []


            infected_mask = self.__population[:, 1] == 1
            infected_indices = np.nonzero(infected_mask)[0]

            infected = self.__pop_matrix[infected_mask]
            num_infected = np.sum(infected)

            successful_contacts_mask =(self.__rstate.poisson(
                infected) >= 1) # len(infected)

            # [len(infected), pop_size]

            successful_contacts_indices = np.nonzero(successful_contacts_mask)

            num_succesful_contacts = np.sum(successful_contacts_mask)

            contact_strength = self.__intense_pdf(num_succesful_contacts)
            infection_prob = self.__infect.pdf(contact_strength)

            newly_infected_mask = self.__rstate.binomial(1, infection_prob)
            # length: num_succesful_contacts
            newly_infected_mask = np.asarray(newly_infected_mask, bool)

            newly_infected_indices = successful_contacts_indices[1][newly_infected_mask]
            
            newly_infected_mask_full = np.zeros(population_size, dtype=bool)
            newly_infected_mask_full[newly_infected_indices] = True

            # recovered and infected people cannot be infected
            # again
            newly_infected_mask_full = (
                newly_infected_mask_full &
                (~(self.__population[:, 1]==1)) &
                (~(self.__population[:, 3]==1))
                )

            num_newly_infected = np.sum(newly_infected_mask_full)

            # adjusting infection duration


            self.__population[infected_indices, 2] -= 1


            recovered_indices = infected_indices[self.__population[infected_indices, 2] == 0]
            # Set recovered
            self.__population[recovered_indices, 1] = 0
            # Set immune
            self.__population[recovered_indices, 3] = 1

            # add new infections

            tmp_dur = np.around(
                self.__infect.pdf_duration(num_newly_infected))

            self.__population[newly_infected_mask_full, 1] = 1
            self.__population[newly_infected_mask_full, 2] = tmp_dur


            if self.__config['save population']:
                self.__distribution.append(np.copy(
                    self.__population
                ))
                self.__total_infections.append(
                    np.sum(self.__fetch_infected()))
                self.__new_infections.append(num_newly_infected)
                self.__immune.append(
                    np.sum(self.__fetch_immune()))


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
        return self.__rstate.uniform(low=0., high=1., size=contacts)

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
        return self.__population[:, 1] == 1

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
        return self.__population[:, 3] == 1
        