import math
from typing import List

import numpy as np
import overpy
import pyproj
from scipy import interpolate

from geopy.distance import geodesic
from utils.interpcurve import bspline


class OSMTrack:
    def __init__(self, ways: List[overpy.Way], identifier=None, name=None, color=None, use_interp=True):
        self.identifier = identifier
        self.name = name
        self.ways = ways
        self.track_nodes = [way.nodes for way in ways]
        self.lon: [float] = []
        self.lat: [float] = []
        self.stitch_ways_to_track()
        self.interp = use_interp

        self.x, self.y = self.convert_lat_lon_to_xy(self.lon, self.lat)
        self.s, self.ds = self.compute_distance_from_lon_lat(self.lon, self.lat)
        self.c = self.compute_curvature(self.x, self.y)

        # Interpolated values
        if self.lon and self.lat:
            tck, u = interpolate.splprep([[float(lon) for lon in self.lon], [float(lat) for lat in self.lat]], s=0)
            u_new = np.arange(0, 1.00, 0.0005)
            out = interpolate.splev(u_new, tck)
            self.lon_interp, self.lat_interp = list(out[0]), list(out[1])
            self.x_interp, self.y_interp = self.convert_lat_lon_to_xy(self.lon_interp, self.lat_interp)
            self.s_interp, self.ds_interp = self.compute_distance_from_lon_lat(self.lon_interp, self.lat_interp)
            self.c_interp = self.compute_curvature(self.x_interp, self.y_interp)

        self.offset = 0
        self.color = color

    def interpolate(self, x, y):
        if len(x) > 0:
            interpolatedTrack = bspline(np.array([x, y]), n=10000)
            return interpolatedTrack[:, 0], interpolatedTrack[:, 1]

        else:
            print('got empty values for x and y')
            return None, None

    def reverse_track(self):
        self.lon.reverse()
        self.lat.reverse()
        self.x.reverse()
        self.y.reverse()

        self.s, self.ds = self.compute_distance_from_lon_lat(self.lon, self.lat)
        self.c = self.compute_curvature(self.x, self.y)

        self.lon_interp.reverse()
        self.lat_interp.reverse()
        self.x_interp.reverse()
        self.y_interp.reverse()

        self.s_interp, self.ds_interp = self.compute_distance_from_lon_lat(self.lon_interp, self.lat_interp)
        self.c_interp = self.compute_curvature(self.x_interp, self.y_interp)

    def flip_curvature(self):
        self.c = [el * -1 for el in self.c]
        self.c_interp = [el * -1 for el in self.c_interp]

    def stitch_ways_to_track(self):
        for i, nodes in enumerate(self.track_nodes[:-1]):
            c_nodes = self.track_nodes[i]
            n_nodes = self.track_nodes[i + 1]

            if c_nodes[-1] == n_nodes[0]:
                pass
            elif c_nodes[0] == n_nodes[-1]:
                c_nodes.reverse()
                n_nodes.reverse()
            elif c_nodes[-1] == n_nodes[-1]:
                n_nodes.reverse()
            elif c_nodes[0] == n_nodes[0]:
                c_nodes.reverse()

        for nodes in self.track_nodes:
            for node in nodes:
                if node.lon not in self.lon and node.lat not in self.lat:
                    self.lon.append(node.lon)
                    self.lat.append(node.lat)

    @staticmethod
    def compute_curvature(x: [float], y: [float]):
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
    def compute_distance_from_xy(x: [float], y: [float]):
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
    def compute_distance_from_lon_lat(lon: [float], lat: [float]):
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
    def convert_lat_lon_to_xy(lon: [float], lat: [float]):
        if lon and lat:
            P = pyproj.Proj(proj='utm', zone=1, ellps='WGS84', preserve_units=True)
            x, y = P(lon, lat)
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
