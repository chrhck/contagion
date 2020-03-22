"""
Name: fd_roll_dice.py
Authors: Stephan Meighen-Berger, Martina Karl
Runs a monte-carlo (random walk) simulation
for the organism interactions.
"""

"Imports"
from sys import exit
import numpy as np
from time import time
from scipy.stats import binom
from fd_config import config

class fd_roll_dice(object):
    """
    class: fd_roll_dice
    Monte-carlo simulation for the light
    emissions.
    Parameters:
        -pdf vel:
            The velocity distribution
        -pdf r:
            The interaction range distribution
        -pdf gamma:
            The photon count emission distribution
        -float current_vel:
            The current velocity used in the shear strength
            calculation
        -int pop:
            The population
        -float regen:
            The regeneration factor
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

    def __init__(self, vel, r, gamma,
                 current_vel,
                 pop, regen, world, log,
                 dt=1., t=np.arange(0., 100., 1.)):
        """
        class: fd_roll_dice
        Initializes the class.
        Parameters:
            -pdf vel:
                The velocity distribution
            -pdf r:
                The interaction range distribution
            -pdf gamma:
                The photon count emission distribution
            -float current_vel:
                The current velocity used in the shear strength
                calculation
            -int pop:
                The population
            -float regen:
                The regeneration factor
            -obj world:
                The constructed world
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
        self.__vel = vel
        self.__curr_vel = current_vel
        self.__pop = pop
        self.__world = world
        self.__regen = regen
        self.__dt = dt
        self.__t = t
        if (self.__pop / self.__world.volume) < config['encounter density']:
            self.__log.debug('Encounters are irrelevant!')
            self._bool_enc = False
        else:
            self.__log.debug('Encounters are relevant!')
            self._bool_enc = True
        # An organism is defined to have:
        #   - dim components for position
        #   - dim components for velocity
        #   - 1 component encounter radius
        #   - 1 component total possible light emission
        #   - 1 component current energy (possible light emission)
        # Total components: dim*dim + 3
        self.__dim = self.__world.dimensions
        self.__dimensions = config['dimensions']*2 + 3
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
        radii = r(pop)
        # The maximum possible light emission is random
        max_light = np.abs(gamma(pop))
        # Giving the population the properties
        self.__population[:, 0:self.__dim] = positions
        self.__population[:, self.__dim:self.__dim*2] = veloc
        self.__population[:, self.__dim*2] = radii
        self.__population[:, self.__dim*2+1] = max_light
        self.__population[:, self.__dim*2+2] = max_light
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
        self.__photon_count = []
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
            # Checking if encounters are relevant
            if self._bool_enc:
                # They are
                encounter_arr = self.__encounter(self.__population[:, 0:self.__dim],
                                                self.__population[:, self.__dim*2])
                # Encounters per organism
                # Subtracting one due to diagonal
                encounters_org = (np.sum(
                    encounter_arr, axis=1
                ) - 1)
                # Encounter emission
                encounter_emission = (
                    encounters_org * self.__population[:, self.__dim*2+1] * 0.1
                )
            else:
                # They are not
                encounter_emission = np.zeros(self.__pop)
            # Light from shearing
            # Vector showing which organisms fired and which didn't
            sheared_number = self.__count_sheared_fired(velocity=self.__curr_vel)
            # Their corresponding light emission
            sheared = self.__population[:, self.__dim*2+1] * 0.1 * sheared_number
            # Total light emission
            light_emission = encounter_emission + sheared
            light_emission = np.array([
                light_emission[idIt]
                if light_emission[idIt] < self.__population[:, self.__dim*2+2][idIt]
                else
                self.__population[:, self.__dim*2+2][idIt]
                for idIt in range(self.__pop)
            ])
            end_enc = time()
            # Subtracting energy
            self.__population[:, self.__dim*2+2] = (
                self.__population[:, self.__dim*2+2] - light_emission
            )
            # Regenerating
            self.__population[:, self.__dim*2+2] = np.array([
                self.__population[:, self.__dim*2+2][i] +
                self.__regen * self.__population[:, self.__dim*2+1][i] * self.__dt
                if (
                    (self.__population[:, self.__dim*2+2][i] +
                     self.__regen * self.__population[:, self.__dim*2+1][i] *
                     self.__dt) <
                    self.__population[:, self.__dim*2+1][i]
                )
                else
                self.__population[:, self.__dim*2+1][i]
                for i in range(self.__pop)
            ])
            # The photon count
            # Assuming 0.1 of total max val is always emitted
            self.__photon_count.append(
                [
                    np.sum(light_emission),
                    np.sum(encounter_emission),
                    np.sum(sheared)
                    ]
            )
            if config['save population']:
                self.__distribution.append(np.copy(
                    self.__population)
                )
            if step % (int(len(self.__t)/10)) == 0:
                self.__log.debug('In step %d' %step)
                self.__log.debug(
                    'Position update took %f seconds' %(end_pos-start_pos)
                )
                self.__log.debug(
                    'Velocity update took %f seconds' %(end_vel-start_vel)
                )
                self.__log.debug(
                    'Emission update took %f seconds' %(end_enc-start_enc)
                )

    @property
    def photon_count(self):
        """
        function: photon_count
        Fetches the photon_count
        Parameters:
            -None
        Returns:
            -photon_count
        """
        return np.array(self.__photon_count)

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

    def __count_sheared_fired(self, velocity=None):
        """
        function: __count_sheared_fired
        Parameters:
            optional float velocity:
                Mean velocity of the water current in m/s
        Returns:
            np.array res:
                Number of cells that sheared and fired.
        """
        # Generating vector with 1 for fired and 0 for not
        # TODO: Step dependence
        res = binom.rvs(
            1,
            self.__cell_anxiety(velocity) * self.__dt,
            size=self.__pop
        )
        return res

    def __cell_anxiety(self, velocity=None):
        """
        function: __cell_anxiety
        Estimates the cell anxiety with alpha * ( shear_stress - min_shear_stress).
        We assume the shear stress to be in the range of 0.1 - 2 Pa and the minimally required shear stress to be 0.1.
        Here, we assume 1.1e-2 for alpha. alpha and minimally required shear stress vary for each population
        Parameters:
            -optional float velocity:
                The velocity of the current in m/s
        Returns:
            -float res:
                Estimated value for the cell anxiety depending of the velocity and thus the shearing
        """
        min_shear = 0.1
        if velocity:
            # just assume 10 percent of the velocity to be transferred to shearing. Corresponds to shearing of
            # 0.01 - 1 Pascal
            shear_stress = velocity * 0.1
        else:
            # Standard velocity is 5m/s
            shear_stress = 0.5

        if shear_stress < min_shear:
            return 0.

        return 1.e-2 * (shear_stress - min_shear)
