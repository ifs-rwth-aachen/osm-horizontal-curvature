from typing import List

import overpy
import pandas as pd


class ReferenceTrack:
    def __init__(self, df: pd.DataFrame):
        self.identifier = df["n"][0]
        self.data = df.drop(["n"], axis=1)
        self.coords = list(zip(self.data.lon, self.data.lat))
        self.name = None
        self.reverse = False
        self.flip_curvature = False
        self.color = None

        self.way_ids = []
        self.ways: List[overpy.Way] = []
        pass

    @property
    def way_ids(self):
        return self._way_ids

    @way_ids.setter
    def way_ids(self, value: []):
        self._way_ids = [int(v) for v in value]

    def __repr__(self):
        return "ReferenceTrack | Identifier: %s, Entries: %d, Len %.2f m, Identifier: %s" % (self.identifier,
                                                                                             len(self.data),
                                                                                             self.data["s"].max(),
                                                                                             str(self.identifier))



