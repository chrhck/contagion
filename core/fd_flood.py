"""
Name: fd_flood.py
Authors: Stephan Meighen-Berger
Filtering algorithms for the organisms
created by dob_vita.
"""

"Imports"
from sys import exit
import numpy as np
from fd_config import config

class fd_flood(object):
    """
    class: fd_flood
    Filters the created organisms.
    This will simplify the fitting.
    Only the worthy survive!
    Parameters:
        -dic life:
            The organisms created
        -obj log:
            The logger
    Returns:
        -None
    """
    def __init__(self, life, org_filter, log):
        """
        function: __init__
        Initializes the flood
        Parameters:
            -dic life:
                The organisms created
            -str org_filter:
                The organism filter to use
            -obj log:
                The logger
        Returns:
            -None
        """
        self.__log = log
        self.__evolved = {}
        if org_filter == 'average':
            self.__log.debug('Filtering by averaging.')
            self.__flood_average(life)
        elif org_filter == 'generous':
            self.__log.debug('All species survive.')
            self.__evolved = life
        elif org_filter == 'depth':
            self.__log.debug('All species below %f are removed'
                              %config['depth filter'])
            self.__flood_depth(life)
        else:
            self.__log.error('Filter not recognized! Please check config')
            exit()

    def __flood_average(self, life):
        """
        function: __flood_average
        Filters the phyla by averaging the constituent
        values.
        Parameters:
            -dic life:
                The organisms created
        Returns:
            -None
        """
        for phyla in config['phyla light'].keys():
            if len(config['phyla light'][phyla]) == 0:
                avg_mean = np.mean(life[phyla][1])
                avg_widt = np.mean(life[phyla][2])
                self.__evolved[phyla] = np.array([
                    [phyla], [avg_mean], [avg_widt]
                ], dtype=object)
                self.__log.debug('1 out of %d %s survived the flood'
                                 %(len(life[phyla][1]), phyla))
            else:
                avg_mean = []
                avg_widt = []
                total_count = 0
                for class_name in config['phyla light'][phyla]:
                    avg_mean.append(np.mean(
                        life[phyla + '_' + class_name][1]
                    ))
                    avg_widt.append(np.mean(
                        life[phyla + '_' + class_name][2]
                    ))
                    total_count += len(life[phyla + '_' + class_name][1]) 
                self.__evolved[phyla] = np.array([
                    [phyla],
                    [np.mean(avg_mean)],
                    [np.mean(avg_widt)]
                ], dtype=object)
                self.__log.debug('1 out of %d %s survived the flood'
                                 %(total_count, phyla))

    def __flood_depth(self, life):
        """
        function: __flood_depth
        Filters the species by removing everythin above the
        depth specified in the config file.
        Parameters:
            -dic life:
                The organisms created
        Returns:
            -None
        """
        for key in life.keys():
            self.__evolved[key] = [[], [], [], []]
            for idspecies, _ in enumerate(life[key][0]):
                if life[key][3][idspecies] >= config['depth filter']:
                     #  The name
                     self.__evolved[key][0].append(
                         life[key][0][idspecies]
                     )
                     # The mean emission
                     self.__evolved[key][1].append(
                         life[key][1][idspecies]
                     )
                     # The FWHM
                     self.__evolved[key][2].append(
                         life[key][2][idspecies]
                     )
                     # The depth
                     self.__evolved[key][2].append(
                         life[key][2][idspecies]
                     )
            total_survive = len(self.__evolved[key][0])
            total_pre_flood = len(life[key][0])
            self.__log.debug('%d out of %d %s survived the flood'
                             %(total_survive, total_pre_flood, key))

    @property
    def evolved(self):
        """
        function: evolved
        Getter function for the survivors
        Parameters:
            -None
        Returns:
            -dic evolved:
                The evolved species
        """
        return self.__evolved
