import logging.config
import os
import warnings
from typing import List

import overpy
import yaml
from tqdm import tqdm

from utils.result import QueryResult
from utils.utils import OSMTrack

with open(os.path.join(os.path.dirname(__file__), "logging_conf.yaml"), 'r', encoding="utf8") as f:
    logging_conf = yaml.safe_load(f.read())
    logging.config.dictConfig(logging_conf)

logger = logging.getLogger(__name__)


class Curvy:
    supported_railway_type = ["rail", "tram", "subway", "light_rail"]

    def __init__(self, lon_sw: float, lat_sw: float, lon_ne: float, lat_ne: float, desired_railway_types=None):
        if desired_railway_types is None:
            desired_railway_types = ["rail", "tram", "subway", "light_rail"]
        else:
            for desired in desired_railway_types:
                if desired not in Curvy.supported_railway_type:
                    raise ValueError("Your desired railway type %s is not supported" % desired)

        if lon_sw == lon_ne or lat_sw == lat_ne:
            raise ValueError("Invalid coordinates")

        if not (-90 <= lat_sw <= 90) or not (-90 <= lat_ne <= 90):
            raise ValueError("Lat. value outside valid range")

        if not (-180 <= lon_sw <= 180) or not (-180 <= lon_ne <= 180):
            raise ValueError("Lon. value outside valid range")

        self.overpass_api = overpy.Overpass()

        self.lon_sw = lon_sw
        self.lat_sw = lat_sw
        self.lon_ne = lon_ne
        self.lat_ne = lat_ne

        self.desired_railway_types = desired_railway_types
        self.query_results = {railway_type: QueryResult for railway_type in self.desired_railway_types}
        logger.info("Initialized curvy in region: %f, %f (SW), %f, %f (NE)" % (self.lon_sw,
                                                                               self.lat_sw,
                                                                               self.lon_ne,
                                                                               self.lat_ne))

    def _create_query(self, railway_type: str):
        if railway_type not in Curvy.supported_railway_type:
            raise ValueError("Your desired railway type %s is not supported" % railway_type)

        query = """(node[""" + "railway" + """=""" + railway_type + """](""" + str(self.lat_sw) + """,""" + str(
            self.lon_sw) + """,""" + str(self.lat_ne) + """,""" + str(self.lon_ne) + """);
                     way[""" + "railway" + """=""" + railway_type + """](""" + str(self.lat_sw) + """,""" + str(
            self.lon_sw) + """,""" + str(self.lat_ne) + """,""" + str(self.lon_ne) + """);
                     relation[""" + "railway" + """=""" + railway_type + """](""" + str(self.lat_sw) + """,""" + str(
            self.lon_sw) + """,""" + str(self.lat_ne) + """,""" + str(self.lon_ne) + """);
                     );
                     (._;>;);
                     out body;
                    """

        return query

    def download_track_data(self, railway_type=None):
        if railway_type:
            # Download data for given railway type
            query = self._create_query(railway_type=railway_type)
            self.query_results[railway_type] = QueryResult(self.overpass_api.query(query), railway_type)
        else:
            # Download data for all desired railway type
            for railway_type in tqdm(self.desired_railway_types):
                query = self._create_query(railway_type=railway_type)
                for attempt in range(3):
                    try:
                        self.query_results[railway_type] = QueryResult(self.overpass_api.query(query), railway_type)
                        break
                    except overpy.exception.OverpassTooManyRequests as e:
                        logging.warning("OverpassTooManyRequest, retrying".format(e))
                    except overpy.exception.OverpassGatewayTimeout as e:
                        logging.warning("OverpassTooManyRequest, retrying".format(e))
                    except overpy.exception.OverpassBadRequest as e:
                        logging.warning("OverpassTooManyRequest, retrying".format(e))
                else:
                    raise RuntimeError("Could download OSM data via Overpass after %d tries." % 3)

        pass

    def query_overpass(self, query: str):
        result = self.overpass_api.query(query)
        return result

    def search_curvy_result(self, way_ids: List[int], railway_type="tram"):
        ways = []

        for way_id in way_ids:
            for way in self.query_results[railway_type].result.ways:
                if way_id == way.id:
                    ways.append(way)

        return ways

    @property
    def lon_sw(self):
        return self._lon_sw

    @lon_sw.setter
    def lon_sw(self, value: float):
        if -180 <= value <= 180:
            self._lon_sw = value
        else:
            warnings.warn("You are trying to set non plausible longitude value %f, keeping existing value"
                          % self.lon_sw, UserWarning)

    @property
    def lat_sw(self):
        return self._lat_sw

    @lat_sw.setter
    def lat_sw(self, value: float):
        if -90 <= value <= 90:
            self._lat_sw = value
        else:
            warnings.warn("You are trying to set non plausible latitude value %f, keeping existing value"
                          % self.lat_sw, UserWarning)

    @property
    def lon_ne(self):
        return self._lon_ne

    @lon_ne.setter
    def lon_ne(self, value: float):
        if -180 <= value <= 180:
            self._lon_ne = value
        else:
            warnings.warn("You are trying to set non plausible longitude value %f, keeping existing value"
                          % self.lon_ne, UserWarning)

    @property
    def lat_ne(self):
        return self._lat_ne

    @lat_ne.setter
    def lat_ne(self, value: float):
        if -90 <= value <= 90:
            self._lat_ne = value
        else:
            warnings.warn("You are trying to set non plausible latitude value %f, keeping existing value"
                          % self.lat_ne, UserWarning)

    def __repr__(self):
        return "curvy | %f, %f (SW), %f, %f (NE)" % (self.lon_sw,
                                                     self.lat_sw,
                                                     self.lon_ne,
                                                     self.lat_ne)
