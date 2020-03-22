"""
Name: fd_adamah.py
Authors: Stephan Meighen-Berger
Constructs the geometry of the system
"""

"Imports"
from sys import exit
from con_config import config
import numpy as np
from scipy import spatial
import pickle

class con_adamah(object):
    """
    class: con_adamah
    Constructs the geometry of the system.
    Parameters:
        -obj log:
            The logger
    Returns:
        -None
    "And a mist was going up from eretz and was watering the whole
     face of adamah."
    """
    # TODO: Organisms will not be evenly distributed in the volume.
    def __init__(self, log):
        """
        function: __init__
        Initializes adamah.
        Parameters:
            -obj log:
                The logger
        Returns:
            -None
        "And a mist was going up from eretz and was watering the whole
        face of adamah."
        """
        self.__log = log
        if config['geometry'] == 'box':
            self.__dim = config['dimensions']
            if not(self.__dim in [2,3]):
                log("Dimensions not supported!")
                exit("Check config file for wrong dimensions!")
            self.__log.debug('Using a box geometry')
            self.__geom_box()
            self.__bounding_box = config['bounding box']
        elif config['geometry'] == 'sphere':
            self.__dim = config['dimensions']
            if not(self.__dim in [2,3]):
                log("Dimensions not supported!")
                exit("Check config file for wrong dimensions!")
            self.__log.debug('Using a sphere geometry')
            self.__geom_sphere()
            self.__bounding_box = config['bounding box']
        elif config['geometry'] == 'custom':
            self.__log.debug("Using custom geometry")
            self.__geom_custom()
        else:
            log.error('Geometry not supported!')
            exit()

    def __geom_box(self):
        """
        function: __geom_box
        Constructs the box geometry
        Parameters:
            -None
        Returns:
            -None
        """
        # The side length of the box
        a = config['box size'] / 2.
        self.__log.debug('The side length is %.1f' %a)
        # The volume of the box
        self.__volume = (a * 2.)**self.__dim
        # The corners of the box
        if self.__dim == 2:
            points = np.array([
                [a, a], [a, -a], [-a, a], [-a, -a]
            ])
        elif self.__dim == 3:
            points = np.array([
                [a, a, -a], [a, -a, -a], [-a, a, -a], [-a, -a, -a],
                [a, a, a], [a, -a, a], [-a, a, a], [-a, -a, a]
            ])
        # The convex hull of the box
        self.__log.debug('Constructing the hull')
        self.__hull = spatial.ConvexHull(points)
        self.__log.debug('Hull constructed')

    def __geom_sphere(self):
        """
        function: __geom_sphere
        Constructs the sphere geometry
        Parameters:
            -None
        Returns:
            -None
        """
        # The side length of the sphere
        r = config['sphere diameter'] / 2.
        self.__log.debug('The radius is %.1f' %r)
        # The volume of the sphere
        if self.__dim == 2:
            self.__volume = r**2 * np.pi
            points = self.__even_circle(config['sphere samples'])
        elif self.__dim == 3:
            self.__volume = (r * 2.)**3. * np.pi * 4./3.
            points = self.__fibonacci_sphere(config['sphere samples'])
        # The corners of the sphere
        points_norm = points / np.linalg.norm(points, axis=1).reshape((len(points), 1))
        points_r = points_norm * r
        # The convex hull of the sphere
        self.__log.debug('Constructing the hull')
        self.__hull = spatial.ConvexHull(points_r)
        self.__log.debug('Hull constructed')

    def __geom_custom(self):
        """
        function: __geom_custom
        Constructs custom geometry from a file
        Parameters:
            -None
        Returns:
            -None
        """
        geom_dic = pickle.load(open(
            "..//data/detector//geometry//" +
            config['custom geometry'],
            'rb')
        )
        self.__log.debug('Constructing the hull')
        self.__hull = spatial.ConvexHull(geom_dic['points'])
        self.__bounding_box = geom_dic['bounding box']
        self.__dim = geom_dic['dimensions']
        self.__volume = geom_dic['volume']
        self.__log.debug('Hull constructed')



    def __fibonacci_sphere(self, samples):
        """
        function: __fibonacci_sphere
        Constructs semi-evenly spread points on a sphere
        Parameters:
            -int samples:
                Number of points
        Returns:
            -np.array points:
                The point cloud
        """
        rnd = 1.
        points = []
        offset = 2./samples
        increment = np.pi * (3. - np.sqrt(5.))
        for i in range(samples):
            y = ((i * offset) - 1) + (offset / 2)
            r = np.sqrt(1 - pow(y,2))
            phi = ((i + rnd) % samples) * increment
            x = np.cos(phi) * r
            z = np.sin(phi) * r

            points.append([x,y,z])
        points = np.array(points)
        return points

    def __even_circle(self, samples):
        """
        function: __even_circle
        Evenly distributes points on a circle
        Parameters:
            -int samples:
                Number of points
        Returns:
            -np.array points:
                The point cloud
        """
        t = np.linspace(0., np.pi*2., samples)
        x = np.cos(t)
        y = np.sin(t)
        points = np.array([
            [x[i], y[i]]
            for i in range(len(x))
        ])
        return points


    @property
    def volume(self):
        """
        function: volume
        Returns the volume
        Parameters:
            -None
        Returns:
            -float volume:
                The volume
        """
        return self.__volume

    @property
    def hull(self):
        """
        function: hull
        Returns the hull
        Parameters:
            -None
        Returns:
            -spatial object hull:
                The hull
        """
        return self.__hull

    @property
    def bounding_box(self):
        """
        function: bounding_box
        Returns the bounding_box size
        Parameters:
            -None
        Returns:
            -float bounding_box:
                The size of the box
        """
        return self.__bounding_box

    @property
    def dimensions(self):
        """
        function: dimensions
        Returns the dimensions
        Parameters:
            -None
        Returns:
            -int dimensions:
                The dimensions of the geometry
        """
        return self.__dim

    def point_in_wold(self, point, tolerance=1e-12):
        """
        function point_in_world
        Checks if the point is in the constructed volume.
        Parameters:
            -np.array point:
                The point to check
            -optional tolerance:
                The allowed tolerance of the search
        Returns:
            bool:
                True or False
        """
        return all(
            (np.dot(eq[:-1], point) + eq[-1] <=tolerance)
            for eq in self.__hull.equations
        )
