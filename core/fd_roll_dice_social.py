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
from scipy.stats import binom
from scipy.stats import norm
from numpy.random import choice
from fd_config import config

class fd_roll_dice_social(object):
    """
    class: fd_roll_dice_social
    Monte-carlo simulation for the light
    emissions.
    Parameters:
        -mean vel:
            The mean social velocity
        -mean r:
            The mean interaction range
        -int pop:
            The population
        -int infected:
            The number of infected people
        -obj log:
            The logger
        -float dt:
            The chosen time step
        -np.array t:
            The time array
    Returns:
        -None
    "God does not roll dice!"
    """

    def __init__(self, vel, vel_var, r, r_var,
                 pop, infected, world, log,
                 dt=1., t=np.arange(0., 100., 1.)):
        """
        function: __init__
        Initializes the class.
        Parameters:
            -float vel:
                The mean social velocity
            -float vel_var:
                The velocity variance
            -float r:
                The mean interaction range
            -float r_var:
                The interaction range variance
            -int pop:
                The population
            -int infected:
                The number of infected people
            -obj log:
                The logger
            -float dt:
                The chosen time step
            -np.array t:
                The time array
        Returns:
            -None
        """
        self.__log = log
        self.__vel_mean = vel
        self.__vel_var = vel_var
        self.__r_mean = r
        self.__r_var = r_var
        if config['pdf move'] == 'gauss':
            self.__vel = self.__vel_distr_norm
            self.__r = self.__r_distr_norm
        else:
            self.__log.error('Unrecognized movement distribution!')
            exit('Check the movement distribution in the config file!')
        self.__pop = pop
        self.__world = world
        self.__dt = dt
        self.__t = t
        # An organism is defined to have:
        #   - dim components for position
        #   - dim components for velocity
        #   - 1 component encounter radius
        #   - 1 interacted or not
        # Total components: dim*dim + 2
        self.__dim = self.__world.dimensions
        self.__dimensions = config['dimensions']*2 + 2
        self.__population = np.zeros((pop, self.__dimensions))
        # Random starting position
        # TODO: Optimize this
        positions = []
        while len(positions) < pop:
            inside = True
            while inside:
                point = np.random.uniform(low=-self.__world.bounding_box/2.,
                                          high=self.__world.bounding_box/2.,
                                          size=self.__dim)
                inside = not(self.__world.point_in_wold(point))
            positions.append(point)
        positions = np.array(positions)
        # Random starting velocities
        veloc = self.__vel(pop).reshape((pop, 1)) * self.__random_direction(pop)
        # Random encounter radius
        radii = self.__r(pop)
        # Sick individuals
        sick_id = choice(pop, size=infected, replace=False)
        bool_array = np.array([
            1
            if i in sick_id
            else
            0
            for i in range(pop)
        ])
        # Giving the population the properties
        self.__population[:, 0:self.__dim] = positions
        self.__population[:, self.__dim:self.__dim*2] = veloc
        self.__population[:, self.__dim*2] = radii
        # Infecting
        self.__population[:, self.__dim*2+1] = bool_array
        if config['save population']:
            self.__log.debug("Saving the distribution of organisms")
            self.__distribution = []
            self.__distribution.append(np.copy(
                self.__population
            ))
        # Running the simulation
        start = time()
        self.__simulation()
        end = time()
        self.__log.debug('MC simulation took %f seconds' % (end-start))

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
        for step, _ in enumerate(self.__t):
            start_pos = time()
            # Updating position
            tmp = (
                self.__population[:, 0:self.__dim] +
                self.__population[:, self.__dim:self.__dim*2] * self.__dt
            )
            # If outside box stay put
            # TODO: Generalize this
            self.__population[:, 0:self.__dim] = np.array([
                tmp[idIt]
                if self.__world.point_in_wold(tmp[idIt])
                else
                tmp[idIt] - self.__population[:, self.__dim:self.__dim*2][idIt] * self.__dt
                for idIt in range(self.__pop)
            ])
            end_pos = time()
            # Updating velocity
            start_vel = time()
            self.__population[:, self.__dim:self.__dim*2] = (
                self.__vel(self.__pop).reshape((self.__pop, 1)) *
                self.__random_direction(self.__pop)
            )
            end_vel = time()
            # Creating encounter array
            start_enc = time()
            encounter_arr = self.__encounter(self.__population[:, 0:self.__dim],
                                            self.__population[:, self.__dim*2])
            # Checking if encounter with infected
            infection_arr = np.array([
                1 if np.any(
                    np.isin(
                        np.nonzero(encounter_arr[i]),
                        np.nonzero(self.__population[:, self.__dim*2+1])
                    )
                )
                else
                0
            for i in range(len(encounter_arr))])
            # Can't re-infect oneself
            infection_arr = np.array([
                1 
                if (
                    (infection_arr[i] == 1) and
                    (self.__population[:, self.__dim*2+1][i] == 0)
                )
                else
                0
            for i in range(len(infection_arr))
            ])
            end_enc = time()
            # Adding to population
            self.__population[:, self.__dim*2+1] += infection_arr
            if config['save population']:
                self.__distribution.append(np.copy(
                    self.__population)
                )
            # Counting new infections
            self.__infections.append(infection_arr)
            if step % (int(len(self.__t)/10)) == 0:
                self.__log.debug('In step %d' %step)
                self.__log.debug(
                    'Position update took %f seconds' %(end_pos-start_pos)
                )
                self.__log.debug(
                    'Velocity update took %f seconds' %(end_vel-start_vel)
                )
                self.__log.debug(
                    'Encounter update took %f seconds' %(end_enc-start_enc)
                )

    @property
    def infections(self):
        """
        function: infections
        Fetches the infection count
        Parameters:
            -None
        Returns:
            -photon_count
        """
        return np.array(self.__infections)

    @property
    def population(self):
        """
        function: population
        Fetches the population
        Parameters:
            -None
        Returns:
            -photon_count
        """
        return np.array(self.__population)
    
    @property
    def distribution(self):
        """
        function: distribution
        Fetches the population distribution
        Parameters:
            -None
        Returns:
            -photon_count
        """
        if config['save population']:
            return np.array(self.__distribution)
        else:
            self.__log.error("Distribution was not saved!")
            exit("Rerun with 'save population' set to True in config file!")

    def __random_direction(self, pop):
        """
        function: __random_direction
        Generates a random direction for
        the velocities of the population
        Parameters:
            -int pop:
                Size of the population
        Returns:
            -np.array direc:
                Array of normalized random
                directions.
        """
        # Creating the direction vector
        # with constraints
        # Generates samples until criteria are acceptable
        direc = []
        # TODO: Optimize this
        for pop_i in range(pop):
            # Sample until angle is acceptable
            angle = (config['angle change'][0] + config['angle change'][1]) / 2.
            while (angle > config['angle change'][0] and angle < config['angle change'][1]):
                new_vec = np.random.uniform(low=-1., high=1., size=self.__dim)
                current_vec = self.__population[pop_i, self.__dim:self.__dim*2]
                # The argument
                arg = (
                    np.dot(new_vec, current_vec) /
                    (np.linalg.norm(new_vec) * np.linalg.norm(current_vec))
                )
                # Making sure parallel and anti-parallel work
                angle = np.rad2deg(np.arccos(
                    np.clip(arg , -1.0, 1.0)
                ))
            direc.append(new_vec)
        direc = np.array(direc)
        # Normalizing
        direc = direc / np.linalg.norm(
            direc, axis=1
        ).reshape((pop, 1))
        return direc

    def __encounter(self, population, radii):
        """
        function: __encounter
        Checks the number of encounters
        Parameters:
            -np.array population:
                The positions of the organisms
            -np.array radii:
                Their encounter radius
        Returns:
            -int num_encounter:
                The number of encounters
        """
        distances = (
            np.linalg.norm(
                population -
                population[:, None], axis=-1
                )
        )
        encounter_arr = np.array([
            distances[idLine] < radii[idLine]
            for idLine in range(0, len(distances))
        ])
        return encounter_arr

    def __vel_distr_norm(self, n):
        """
        function: __vel_distr_norm:
        Gaussian velocity distribution
        Parameters:
            -int n:
                The sample size
        """
        return(norm.rvs(size=n,
                        loc=self.__vel_mean,
                        scale=self.__vel_var))

    def __r_distr_norm(self, n):
        """
        function: __r_distr_norm:
        Gaussian radii distribution
        Parameters:
            -int n:
                The sample size
        """
        encounter_r = []
        for i in range(n):
            res = -1
            while res <=0:
                res = norm.rvs(
                    loc=self.__r_mean,
                    scale=self.__r_var
                )
            encounter_r.append(res)
        return np.array(encounter_r)