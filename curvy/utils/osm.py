import itertools
import math
from typing import List


class OSMRelation:
    def __init__(self, id, ways: List[overpy.Way], name: str = None, color=None):
        """ Class Representing an OpenStreetMap relation

        Parameters
        ----------
        id: int
            ID of the relation
        ways: List[overpy.Way]
            Ways part of the relation
        name: str
            Name of the relation
        color:
            Color when used to draw the track e.g. using ipyleaflet
        """
        self.id = id
        self.name = name
        self.ways = ways
        self.way_nodes = [way.nodes for way in ways]  # List of list of nodes
        self.nodes = list(itertools.chain.from_iterable(self.way_nodes))  # list of nodes
        self.lon: [float] = []
        self.lat: [float] = []

        self.stitch_ways_to_track()

        self.x, self.y = self.convert_lon_lat_to_xy(self.lon, self.lat)
        self.s, self.ds = self.compute_distance_from_lon_lat(self.lon, self.lat)
        self.c = self.compute_curvature(self.x, self.y)

        self.offset = 0
        self.color = color

    def reverse_track(self):
        """
            Reverses the lon/lat respectively x/y coordinates
        """
        self.lon.reverse()
        self.lat.reverse()
        self.x.reverse()
        self.y.reverse()

        self.s, self.ds = self.compute_distance_from_lon_lat(self.lon, self.lat)
        self.c = self.compute_curvature(self.x, self.y)

    def flip_curvature(self):
        """
            Flips the calculated curvature upside down
        """
        self.c = [el * -1 for el in self.c]

    def stitch_ways_to_track(self):
        """
            The order of the nodes inside a way can vary. Hence, nodes cannot be put into a list directly.
            This prototype algorithm tries to stitch the nodes of all ways together to a single track.
        """
        for i, nodes in enumerate(self.way_nodes[:-1]):
            c_nodes = self.way_nodes[i]
            n_nodes = self.way_nodes[i + 1]

            if c_nodes[-1] == n_nodes[0]:
                pass
            elif c_nodes[0] == n_nodes[-1]:
                c_nodes.reverse()
                n_nodes.reverse()
            elif c_nodes[-1] == n_nodes[-1]:
                n_nodes.reverse()
            elif c_nodes[0] == n_nodes[0]:
                c_nodes.reverse()

        for nodes in self.way_nodes:
            for node in nodes:
                if node.lon not in self.lon and node.lat not in self.lat:
                    self.lon.append(node.lon)
                    self.lat.append(node.lat)

    def to_ipyleaflet(self):
        """ Converts the coordinates to the format required by ipyleaflet for drawing
        Returns
        -------
            list
        """
        if self.lat and self.lon:
            return [[float(lat), float(lon)] for lat, lon in zip(self.lat, self.lon)]
        else:
            return [[]]

    @staticmethod
    def compute_curvature(x: List[float], y: List[float]) -> List[float]:
        """ Calculates the Menger curvature for a set of coordinates

        Parameters
        ----------
        x
        y

        Returns
        -------
        list
        """
        if len(x) != len(y):
            raise ValueError("x and y have to be same length")

        if len(x) > 0 and len(y) > 0:
            c = [0]

            for i in range(len(x)):
                if not (i == 0 or i == len(x) - 1):
                    # Get three neighbored points
                    x1 = x[i - 1]
                    y1 = y[i - 1]

                    x2 = x[i]
                    y2 = y[i]

                    x3 = x[i + 1]
                    y3 = y[i + 1]

                    # Get distance between each of the points
                    s_a = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    s_b = math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2)
                    s_c = math.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)

                    s = (s_a + s_b + s_c) / 2
                    a = s * (s - s_a) * (s - s_b) * (s - s_c)

                    if a > 0:
                        A = math.sqrt(a)
                    else:
                        A = 0

                    # Calculate sign
                    dx12 = x2 - x1
                    dy12 = y2 - y1

                    dx23 = x3 - x2
                    dy23 = y3 - y2

                    sgn = np.sign(np.cross([dx12, dy12], [dx23, dy23]))

                    # Menger Curvature
                    res = sgn * 4 * A / (s_a * s_b * s_c)
                    if not math.isnan(res):
                        c.append(res)
                    else:
                        c.append(0.0)

            c.append(0)
            return c
        else:
            return []

    @staticmethod
    def compute_distance_from_xy(x: List[float], y: List[float]) -> Tuple[list, list]:
        """ Computes distances between a set of x y coordinates as well as the total distance

        Parameters
        ----------
        x
        y

        Returns
        -------
        Tuple
            List containing total distance, list of pairwise distances
        """
        if len(x) != len(y):
            raise ValueError("x and y have to be same length")

        if x and y:
            s = [0]
            ds = []

            for i in range(len(x)):
                if i >= 1:
                    x1 = x[i - 1]
                    y1 = y[i - 1]

                    x2 = x[i]
                    y2 = y[1]

                    ds.append(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))

            # Integrate ds
            s.extend(np.cumsum(ds))

            return s, ds
        else:
            return [], []

    @staticmethod
    def compute_distance_from_lon_lat(lon: List[float], lat: List[float]):
        """ Computes pairwise and total distance using geodesic distance of individual lat/lon coordinates

        Parameters
        ----------
        lon: list
        lat: list

        Returns
        -------
        list, list
        """
        if len(lon) != len(lat):
            raise ValueError("x and y have to be same length")

        if lon and lat:
            s = [0]
            ds = []

            for i in range(len(lon)):
                if i >= 1:
                    ds.append(geodesic((lat[i - 1], lon[i - 1]), (lat[i], lon[i])).meters)

            # Integrate ds
            s.extend(np.cumsum(ds))

            return s, ds
        else:
            return [], []

    @staticmethod
    def convert_lon_lat_to_xy(lon: List[float], lat: List[float]):
        """ Convert lon/lat coordinates to a metric coordinate system. The first coordinate is used for zero-point in
        the metric coordinate system

        Parameters
        ----------
        lon: list[float]
        lat: list[float]

        Returns
        -------
        list, list
        """
        if lon and lat:
            proj = pyproj.Proj(proj='utm', zone=1, ellps='WGS84', preserve_units=True)
            x, y = proj(lon, lat)
            return [el - x[0] for el in x], [el - y[0] for el in y]
        else:
            return [], []

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        if self.s:
            self.s = [el + value for el in self.s]

        self._offset = value

    def to_ipyleaflef(self) -> List[list]:
        """

        Returns
        -------

        """
        if self.nodes:
            logger.warning("Relation has no nodes!")
            return [[n.lat, n.lon] for n in self.nodes]
        else:
            return [[]]

class OSMRailwayLine(OSMRelation):
    def __init__(self, id: int, ways: list, tags: dict, members: list):
        """ Class representing a railway line

        See https://wiki.openstreetmap.org/wiki/Tag:railway%3Drail

        Parameters
        ----------
        id: int
            ID of the railway line
        ways: list
            List of ways of the railway line
        tags: dict
            Tags associated with the railway line
        members: list
            list of nodes and ways associated with the railway line
        """
        super().__init__(id=id, ways=ways, name=tags.get("name", ""),
                         color=tags.get("colour", generate_random_color("HEX")))
        self.__dict__.update(tags)
        self.members = members

    def __repr__(self):
        return self.__dict__.get("name", "")