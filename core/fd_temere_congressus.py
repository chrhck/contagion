"""
Name: fd_temere_congressus.py
Authors: Stephan Meighen-Berger
Models random encounters of organisms
"""

"Imports"
import numpy as np
from sys import exit
from scipy.stats import norm
import csv
from fd_config import config

class fd_temere_congressus(object):
    """
    class: fd_temere_congressus
    Collection of interaction models between
    the organisms.
    Parameters:
        -obj log:
            The logger
    Returns:
        -None
    """
    def __init__(self, log):
            """
            function: __init__
            Initializes random encounter models of
            the organisms.
            Parameters:
                -obj log:
                    The logger
            Returns:
                -None
            """
            self.__log = log
            # Storage of the movement patterns
            self.__move = {}
            # The model
            if config['encounter'] == "Gerritsen-Strickler":
                self.__log.info('Using the Gerritsen_Strickler model')
                self.__model = self.__Gerritsen_Strickler
            else:
                self.__log.error("Unknown encounter model! Please check " +
                                "The config file!")
                exit('Set encounter model is wrong!')
            # Distribution parameters:
            self.__encounter_params()
            # The distributions
            if config['pdf move'] == 'gauss':
                self.__log.debug('Construction gaussian' +
                                ' distributions for movement')
                # Velocity distr
                self.__vel_distr = self.__vel_distr_norm
                # Velocity mean
                self.__vel_mean = self.__distr_par[0]
                # Radius distr
                self.__r_distr = self.__r_distr_norm
                # Radius mean
                self.__r_mean = self.__distr_par[2]
                # Emission distr
                self.__gamma_distr = self.__gamma_distr_norm
                # Emission mean
                self.__gamma_mean = self.__distr_par[4]
                self.__log.debug('Finished gaussian' +
                                ' distributions for movement')
            else:
                self.__log.error('Unrecognized movement distribution!')
                exit('Check the movement distribution in the config file!')

    @property
    def pdf(self):
        """
        function: pdf
        Fetches the pdfs
        Parameters:
            -None
        Returns:
            -None
        """
        return [
            self.__vel_distr,
            self.__r_distr,
            self.__gamma_distr
        ]

    def rate(self, v, n , volume):
        """
        function: rate
        Calculates the encounter rate for a single
        point like organism.
        Parameters:
            -float v:
                The velocity of the organism
            -float n:
                The number of background organisms
            -float volume:
                The size of the volume of interest
                (in mm^3)
        Returns:
            -flaot Z:
                The mean number of produced photons
        """
        # Repeating until positive roll
        res = -1.
        while res < 0.:
            # The velocities
            u = self.__vel_distr(n)
            # The radii
            R = self.__r_distr(n)
            # The light yield
            ly = self.__gamma_distr(n)
            res = np.array([
                self.__model(v, n, volume, u[itera], R[itera], ly[itera])
                for itera in range(0, n)
            ]).mean()
        return res * n

    def rate_avg(self, v, n , volume):
        """
        function: rate_avg
        Calculates the average encounter rate for a single
        point like organism.
        Parameters:
            -float v:
                The velocity of the organism
            -float n:
                The number of background organisms
            -float volume:
                The size of the volume of interest
                (in mm^3)
        Returns:
            -flaot Z:
                The mean number of produced photons
        """
        # The velocity
        u = self.__vel_mean
        # The radius
        R = self.__r_mean
        # The light yield
        ly = self.__gamma_mean
        res = self.__model(v, n, volume, u, R, ly)
        return res * n    

    def __Gerritsen_Strickler(self, v, n, volume, u, R, ly):
        """
        function: __Gerritsen_Strickler
        Calculates the encounter rate for a single
        point like organism.
        Parameters:
            -float v:
                The velocity of the organism
            -float n:
                The number of background organisms
            -float volume:
                The size of the volume of interest
                (in mm^3)
            -float u:
                Velocity of other organisms
            -float R:
                Encounter radius
            -float ly:
                Light production
        Returns:
            -float Z:
                The mean number of produced photons
        """
        # The density
        N_b = n / volume
        if v >= u:
            Z = (
                np.pi * R**2. * N_b / 3. *
                (u**2. + 3. * v**2.) / v
            )
        else:
            Z = (
                np.pi * R**2. * N_b / 3. *
                (v**2. + 3. * u**2.) / u
            )
        return (np.mean(ly * Z))

    def __encounter_params(self):
        """
        function: __encounter_params:
        Constructs parameters for the pdfs
        in the encounter model.
        Parameters:
            -None
        Returns:
            -None
        """
        for phyla in config['phyla move']:
            self.__log.debug('Loading phyla: %s' %phyla)
            self.__log.debug('Loading and parsing %s.txt' %phyla)
            with open('../data/life/movement/%s.txt' %phyla, 'r') as txtfile:
                tmp = list(
                    csv.reader(txtfile, delimiter=',')
                )
                # Converting to numpy array
            tmp = np.asarray(tmp)
            self.__move[phyla] = np.array(
                    [
                        # The name
                        tmp[:, 0].astype(str),
                        # The mean velocity in mm/s
                        tmp[:, 1].astype(np.float32),
                        # The encounter radius
                        tmp[:, 2].astype(np.float32),
                        # Most probable photon count 1e10
                        tmp[:, 3].astype(np.float32)
                    ],
                    dtype=object
                )
        # Distribution parameters:
        self.__distr_par = np.array([
            # Constructing the mean velocity
            np.mean(np.array([
                self.__move[phyla][1].mean()
                for phyla in config['phyla move']
            ])),
            # The velocity variation
            np.mean(np.array([
                self.__move[phyla][1].var()
                for phyla in config['phyla move']
            ])),
            # The mean encounter radius
            np.mean(np.array([
                self.__move[phyla][2].mean()
                for phyla in config['phyla move']
            ])),
            # The radius variation
            np.mean(np.array([
                self.__move[phyla][2].var()
                for phyla in config['phyla move']
            ])),
            # The mean photon count
            np.mean(np.array([
                self.__move[phyla][3].mean()
                for phyla in config['phyla move']
            ])),
            # photon count variation
            np.mean(np.array([
                self.__move[phyla][3].var()
                for phyla in config['phyla move']
            ]))
        ])

    def __vel_distr_norm(self, n):
        """
        function: __vel_distr_norm:
        Gaussian velocity distribution
        Parameters:
            -int n:
                The sample size
        """
        return(norm.rvs(size=n,
                        loc=self.__distr_par[0],
                        scale=self.__distr_par[1]))

    def __r_distr_norm(self, n):
        """
        function: __r_distr_norm:
        Gaussian radii distribution
        Parameters:
            -int n:
                The sample size
        """
        return(norm.rvs(size=n,
                        loc=self.__distr_par[2],
                        scale=self.__distr_par[3]))

    def __gamma_distr_norm(self, n):
        """
        function: __gamma_distr_norm:
        Gaussian photon distribution
        Parameters:
            -int n:
                The sample size
        """
        return(norm.rvs(size=n,
                        loc=self.__distr_par[4],
                        scale=self.__distr_par[5]))
