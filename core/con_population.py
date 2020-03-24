"""
Name: con_population.py
Authors: Stephan Meighen-Berger
Constructs the population.
"""

"Imports"
from sys import exit
import numpy as np
import scipy.stats
from scipy.stats import norm, halfnorm, truncnorm
import scipy.sparse as sparse

class CON_population(object):
    """
    class: CON_pupulation
    Class to help with the construction of a realistic population
    Paremeters:
        -int pop:
            The size of the populations
        -obj log:
            The logger
        -dic config:
            The dictionary from the config file
    Returns:
        -None
    """
    def __init__(self, pop, log, config, rstate=None):
        """
        function: __init__
        Initializes the population
        Parameters:
            -int pop:
                The size of the population
            -obj log:
                The logger
            -dic config:
                The dictionary from the config file
        Returns:
            -None
        """
        self.__config = config
        self.__log = log
        if rstate is None:
            self.__log.warning("No random state given, constructing new state")
            rstate = np.random.RandomState()
        self.__rstate = rstate

        self.__log.info('Constructing social circles for the population')
        self.__log.debug('Number of people in social circles')
        if self.__config['social circle pdf'] == 'gauss':
            self.__social_circles = self.__social_pdf_norm(pop)
        else:
            self.__log.error('Unrecognized social pdf! Set to ' + self.__config['social circle pdf'])
            exit('Check the social circle distribution in the config file!')

        self.__log.debug('The social circle interactions for each person')
        if self.__config['social circle interactions pdf'] == 'gauss':
            self.__sc_interactions = self.__sc_interact_norm(pop)
        else:
            self.__log.error('Unrecognized sc interactions pdf! Set to ' +
                             self.__config['social circle interactions pdf'])
            exit('Check the social circle interactions distribution in the config file!')

        

        self.__log.debug('Constructing population')

        # The probability of a person being connected to another person is 
        # given by population size / social circle size
        p_contact = self.__social_circles / (pop-1)
        """
        interaction_matrix = self.__rstate.binomial(
            1, p_contact, size=(pop, pop))
        """

        interaction_matrix = sparse.lil_matrix((pop, pop), dtype=bool)

        for row, p in enumerate(p_contact):
            interaction_matrix[row] = scipy.stats.binom.rvs(
                1, p_contact, size=pop)

        
        interaction_matrix = sparse.triu(interaction_matrix, 1) +\
            sparse.triu(interaction_matrix.transpose(), 1)
              

        #interaction_matrix[lower_tri_ind] = 0
        interaction_matrix.setdiag(0)
        interaction_matrix = interaction_matrix.asfptype()
  
        
        # Sample the number of interactions per person
        # TODO this could be made dependent on the social circle / contact

        num_contacts = self.__sc_interactions
        num_connections = (interaction_matrix.sum(axis=1)-1)
        num_connections = np.asarray(num_connections).squeeze()

        contact_rate = num_contacts / (num_connections)
        contact_rate[num_connections <= 0] = 0

        d = sparse.spdiags(contact_rate, 0, pop, pop)

        #interaction_matrix = interaction_matrix.tocsr()
        interaction_matrix = d * interaction_matrix
        

        # For each two person encounter (A <-> B) there are now two rates,
        # one from person A and one from B. Pick the max for both.


        upper_triu = sparse.triu(interaction_matrix, 1)
        upper_triu_transp = sparse.triu(
            interaction_matrix.transpose(), 1)
        interaction_matrix = upper_triu.maximum(upper_triu_transp)

        """
        upper_tri_ind = np.triu_indices(pop, 1)
        lower_tri_ind = np.triu_indices(pop, 1)

        interaction_matrix[lower_tri_ind] = 0.5*(
            interaction_matrix[lower_tri_ind] +
            interaction_matrix.T[lower_tri_ind])

        interaction_matrix[upper_tri_ind] = 0
        interaction_matrix[np.diag_indices(pop)] = 1

        
        print(interaction_matrix)
        """
        self.__interaction_matrix = interaction_matrix

        """
        self.__pop = np.array([
            [np.random.choice(range(pop), size=self.__social_circles[i], replace=False),
             self.__sc_interactions[i]]
            for i in range(pop)
        ])
        """

    @property
    def population(self):
        """
        function: population
        Returns the population
        Parameters:
            -None
        Returns:
            -np.array population:
                The constructed population
        """
        # return self.__pop
        return self.__interaction_matrix

    def __social_pdf_norm(self, pop):
        """
        function: __social_pdf_norm
        Constructs the number of people in each person's circle
        Parameters:
            -int pop:
                The population size
            -np.array circles:
                The number of people in each circle
        Returns:
            -np.array circles:
                The size of the social circles for every individual
        """
        
        mean = self.__config['average social circle']
        scale = self.__config['variance social circle']

        # Minimum social circle size is 0
        a, b = (0 - mean) / scale, (pop - mean) / scale

        # could also use binomial here
        circles = truncnorm.rvs(
            a, b, loc=mean, scale=scale, size=pop, random_state=self.__rstate)
        circles = np.asarray(circles, dtype=int)

        """
        for _ in range(pop):
            res = -1
            while res < 0:
                res = int(norm.rvs(
                    size=1,
                    loc=self.__config['average social circle'],
                    scale=self.__config['variance social circle']))
            circles.append(res)
        circles = np.array(circles)
        """
        return circles

    def __sc_interact_norm(self, pop):
        """
        function: __sc_interact_norm
        The number of interactions within each person's circle
        Parameters:
            -int pop:
                The size of the population
        Returns:
            -np.array interact:
                The number of interactions within the social circle
        """

        mean = self.__config['mean social circle interactions']
        scale = self.__config['variance social circle interactions']


        a, b = (0 - mean) / scale, (self.__social_circles - mean) / scale

        zero_friends = b <= a
        b[zero_friends] = (1 - mean) / scale
        # could also use binomial here

        interactions = truncnorm.rvs(
            a, b, loc=mean, scale=scale, size=pop, random_state=self.__rstate)
        interactions = np.asarray(interactions, dtype=int)

        interactions[zero_friends] = 0
        return interactions

        """
        interact = []
        # Re-rolling negative values
        for i in range(pop):
            res = -1
            while res < 0:
                res = int(norm.rvs(
                    size=1,
                    loc=self.__config[''],
                    scale=self.__config['']
                ))
                # Never interact more than the size of the sc
                if res > self.__social_circles[i]:
                    res = -1
            interact.append(res)
        return(np.array(interact))
        """

