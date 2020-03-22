"""
Name: fourth_day.py
Authors: Stephan Meighen-Berger
Main interface to the fourth_day module.
This package calculates the light yields and emission specta
of organisms in the deep sea using a combination of modelling
and data obtained by deep sea Cherenkov telescopes. Multiple
calculation routines are provided.
Notes:
    - Multiple distinct types (Phyla pl., Phylum sg.)
        - Domain: Bacteria
            -Phylum:
                Dinoflagellata
        - Chordate:
            During some period of their life cycle, chordates
            possess a notochord, a dorsal nerve cord, pharyngeal slits,
            an endostyle, and a post-anal tail:
                Subphyla:
                    -Vertebrate:
                        E.g. fish
                    -Tunicata:
                        E.g. sea squirts (invertibrate filter feeders)
                    -Cephalochordata
                        E.g. lancelets (fish like filter feeders)
        - Arthropod:
            Subphyla:
                -Crustacea:
                    E.g. Crabs, Copepods, Krill, Decapods
        - Cnidaria:
            Subphyla:
                -Medusozoa:
                    E.g. Jellyfish
"""

"Imports"
# Native modules
import logging
import numpy as np
from time import time
# Package modules
from fd_config import config
from fd_immaculate_conception import fd_immaculate_conception
from fd_flood import fd_flood
from fd_genesis import fd_genesis
from fd_tubal_cain import fd_tubal_cain
from fd_adamah import fd_adamah
from fd_temere_congressus import fd_temere_congressus
from fd_lucifer import fd_lucifer
from fd_roll_dice import fd_roll_dice
from fd_roll_dice_social import fd_roll_dice_social
from fd_yom import fd_yom

class FD(object):
    """
    class: FD
    Interace to the FD package. This class
    stores all methods required to run the simulation
    of the bioluminescence
    Parameters:
        -str org_filter:
            How to filter the organisms.
    Returns:
        -None
    """
    def __init__(self,
     org_filter=config['filter']
     ):
        """
        function: __init__
        Initializes the class FD.
        Here all run parameters are set.
        Parameters:
            -str org_filter:
                How to filter the organisms.
            -bool monte_carlo:
                Use of monte carlo or not
        Returns:
            -None
        """
        "Logger"
        # Basic config empty for now
        logging.basicConfig()
        # Creating logger user_info
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.DEBUG)
        self.log.propagate = False
        # creating file handler with debug messages
        self.fh = logging.FileHandler('../fd.log', mode='w')
        self.fh.setLevel(logging.DEBUG)
        # console logger with a higher log level
        self.ch = logging.StreamHandler()
        self.ch.setLevel(config['debug level'])
        # Logging formatter
        formatter = logging.Formatter(
            fmt='%(levelname)s: %(message)s'
        )
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)
        # Adding the handlers
        self.log.addHandler(self.fh)
        self.log.addHandler(self.ch)
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Welcome to FD!')
        self.log.info('This package will help you model deep sea' +
                      ' bioluminescence! (And some other things)')
        self.log.info('---------------------------------------------------')
        self.log.info('---------------------------------------------------')
        self.log.info('Checking simulation type')
        self.log.info('Simulation is set to ' + config['simulation type'])
        if config['simulation type'] == 'bioluminescence':
            self.log.info('Bioluminescence simulation run')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            self.log.info('Creating life...')
            # Life creation
            self.life = fd_immaculate_conception(self.log).life
            self.log.info('Creation finished')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            self.log.info('Initializing flood')
            # Filtered species
            self.evolved = fd_flood(self.life, org_filter, self.log).evolved
            self.log.info('Survivors collected!')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            self.log.info('Starting genesis')
            # PDF creation for all species
            self.pdfs = fd_genesis(self.evolved, self.log).pdfs
            self.log.info('Finished genesis')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            self.log.info('Forging combined distribution')
            self.log.info('To use custom weights for the populations, ')
            self.log.info('run fd_smithing with custom weights')
            # Object used to create pdfs
            self.smith = fd_tubal_cain(self.pdfs, self.log)
            # Fetching organized keys
            self.keys = self.smith.keys
            # Weightless pdf distribution
            self.pdf_total = self.smith.fd_smithing()
            self.log.info('Finished forging')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            self.log.info('Creating the world')
            #  The volume of interest
            self.world = fd_adamah(self.log)
            self.log.info('Finished world building')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            self.log.info('Random encounter model')
            # TODO: Make this pythonic
            # TODO: It would make sense to add this to immaculate_conception
            # TODO: Unify movement model with the spectra model
            self.rate_model = fd_temere_congressus(self.log)
            self.log.info('Finished the encounter model')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            self.log.info('To run the simulation use the solve method')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
        # Social distancing
        elif config['simulation type'] == 'social distancing':
            # TODO: Population model
            # PDFs are dealt with by the solver
            self.log.info('Social distancing simulation run')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            self.log.info('Creating the world')
            # TODO: Social circle models for the "world"
            # TODO: Encounter distribution models
            #  The volume of interest
            self.world = fd_adamah(self.log)
            self.log.info('Finished world building')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
        else:
            self.log.error('Unrecognized simulation type!')
            exit('Please check the config file')

    # TODO: Add incoming stream of organisms to the volume
    # TODO: Creat a unified definition for velocities
    def solve(self, population, velocity,
              distances, photon_count,
              seconds=100,
              regen=1e-3,
              dt=config['time step'],
              vel_var=1.,
              dist_var=1.,
              infected=1):
        """
        function: solve
        Calculates the light yields depending on input
        Parameters:
            -float population:
                The number of organisms
            -float velocity:
                The mean velocity of the current in m/s,
                or the mean "social" velocity 
            -float distances:
                The distances to use. For a social run,
                this is the mean infection distance
            -float photon_count:
                The mean photon count per collision.
                Unused in social simulations
            -int seconds:
                Number of seconds to simulate. This is used by
                the mc routines.
            -float regen:
                The regeneration factor
            -float dt:
                The time step to use. Needs to be below 1
            -float vel_var:
                The social velocity variance
            -float dist_var:
                The social distance variance
            -int infected:
                The number of infected people
        Returns:
            -np.array result:
                The resulting light yields
        """
        if dt > 1.:
            self.log.error("Chosen time step too large!")
            exit("Please run with time steps smaller than 1s!")
        # Bioluminescence
        if config['simulation type'] == 'bioluminescence':
            self.log.info('Calculating light yields')
            self.log.debug('Monte-Carlo run')
            # The time grid
            self.t = np.arange(0., seconds, dt)
            # The simulation
            pdfs = self.rate_model.pdf
            # TODO: Update this to take convex hulls.
            # TODO: This will improve the check if point cloud is
            #       inside
            # TODO: Add switch which removes encounter model
            #       depending on density of the organisms
            self.mc_run = fd_roll_dice(
                pdfs[0],
                pdfs[1],
                pdfs[2],
                velocity,
                population,
                regen,
                self.world,
                self.log,
                dt=dt,
                t=self.t
            )
            self.log.debug('---------------------------------------------------')
            self.log.debug('---------------------------------------------------')
            # Applying pulse shapes
            pulses = fd_yom(self.mc_run.photon_count, self.log).shaped_pulse
            self.log.debug('---------------------------------------------------')
            self.log.debug('---------------------------------------------------')
            # The total emission
            self.log.debug('Total light')
            result = fd_lucifer(
                pulses[:, 0],
                distances, self.log
            ).yields * photon_count
            # The possible encounter emission without regen
            self.log.debug('Encounter light')
            result_enc = fd_lucifer(
                pulses[:, 1],
                distances, self.log
            ).yields * photon_count
            # The possible sheared emission without regen
            self.log.debug('Shear light')
            result_shear = fd_lucifer(
                pulses[:, 2],
                distances, self.log
            ).yields * photon_count
            self.log.debug('---------------------------------------------------')
            self.log.debug('---------------------------------------------------')
            self.log.info('Finished calculation')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            return result, result_enc, result_shear
        # Social distancing
        elif config['simulation type'] == 'social distancing':
            self.log.debug('Monte-Carlo run')
            self.t = np.arange(0., seconds, dt)
            self.mc_run = fd_roll_dice_social(
                velocity,
                vel_var,
                distances,
                dist_var,
                population,
                infected,
                self.world,
                self.log,
                dt=dt,
                t=self.t
            )
            self.log.debug('---------------------------------------------------')
            self.log.debug('---------------------------------------------------')
            self.log.info('Finished calculation')
            self.log.info('---------------------------------------------------')
            self.log.info('---------------------------------------------------')
            return self.mc_run.infections
        else:
            self.log.error('Unrecognized simulation type!')
            exit('Please check the config file')