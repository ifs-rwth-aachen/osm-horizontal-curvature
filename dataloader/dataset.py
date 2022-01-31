import json
import math
from scipy import stats
import numpy as np
from typing import Dict

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tikzplotlib
from scipy import interpolate

from dataloader.utils import ReferenceTrack
from utils.utils import OSMTrack
import os

matplotlib.use("Qt5Agg")

str_template = r"""
% This file was created by tikzplotlib v0.9.4.
\begin{tikzpicture}
\begin{axis}[
width=.8\linewidth,height=6cm,
axis line style={white},
tick align=outside,
x grid style={white},
xlabel={$s$ [m]},
xmajorgrids,
xmajorticks=true,
xmin=0, xmax=1500,
xtick style={color=black},
x grid style={gray},
y grid style={gray},
ylabel={$\kappa(s)$ [1/m]},
ymajorgrids,
ymin=-0.05, ymax=0.05,
ytick pos=left,
ytick style={color=black},
ytick={-0.04,-0.03,-0.02,-0.01,6.93889390390723e-18,0.01,0.02,0.03,0.04},
yticklabels={-0.04,-0.03,-0.02,-0.01,0.00,0.01,0.02,0.03,0.04}
]
\addplot[line width=1.2pt, RWTHRot100, opacity=1] table [col sep=comma, x=s, y=curv_hor]
{resultdata/results_track_{name}_ref.csv};
\addplot[line width=1.2pt, RWTHBlau100, opacity=1] table [col sep=comma, x=s, y=curv_hor]
{resultdata/results_track_{name}_osm.csv};
\legend{$\kappa_{ref}$, $\kappa_{OSM}$}
\end{axis}

\begin{axis}[
width=.8\linewidth,height=6cm,
axis line style={white},
axis y line=right,
legend cell align={left},
legend style={fill opacity=0.8, draw opacity=1, text opacity=1, 
draw=white!80!black, fill=white!89.8039215686275!black},
tick align=outside,
x grid style={black},
xmajorticks=false,
xmin=-364.6698, xmax=7658.0658,
xtick style={color=black},
y grid style={white},
ylabel={$R$ [m]},
ymin=0, ymax=1,
ytick pos=right,
ytick style={color=black},
ytick={0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1},
yticklabels={,-25.0,-33.3,-50.0,-100.0,Straight,100.0,50.0,33.3,25.0,}
]
\end{axis}

\end{tikzpicture}
"""


class Dataset:
    def __init__(self, path):
        self.path = path
        self.dataset = self.load_dataset()
        with open("data/mannheim_osm_mappings.json") as f:
            self.mappings = json.load(f)["mappings"]

        self.reference_tracks: Dict[str, ReferenceTrack] = self.convert_to_reference_tracks()
        self.osm_tracks: Dict[str, OSMTrack] = {}
        self.bbox = self.dataset["lon"].min(), self.dataset["lat"].min(), \
                    self.dataset["lon"].max(), self.dataset["lat"].max()  # Min lon, Min lat, Max lon, Max lat

    def load_dataset(self):
        dataset = pd.read_pickle(self.path)
        return dataset

    def convert_to_reference_tracks(self) -> Dict[str, ReferenceTrack]:
        track_names = self.dataset["n"].unique()

        tracks: Dict[str, ReferenceTrack] = {}

        for identifier in track_names:
            track_df = self.dataset[self.dataset["n"] == identifier]
            ref_track = ReferenceTrack(track_df)

            if ref_track.identifier in self.mappings.keys():
                ref_track.name = self.mappings[ref_track.identifier]["name"]
                ref_track.way_ids = self.mappings[ref_track.identifier]["way_ids"]
                ref_track.reverse = self.mappings[ref_track.identifier]["reverse"]
                ref_track.flip_curvature = self.mappings[ref_track.identifier]["flip_curvature"]
                ref_track.color = self.mappings[ref_track.identifier]["color"]

            tracks[identifier] = ref_track

            if ref_track.name:
                tracks[ref_track.name] = ref_track

        return tracks

    def plot_network(self):
        fig, ax = plt.subplots(1, 1)

        asp = 1 / math.cos(math.radians((self.bbox[1] + self.bbox[2]) / 2))

        for key, track in self.reference_tracks.items():
            if "1-S" in key:
                ax.plot(track.data["lon"], track.data["lat"], c="black")

        # Plot OSM coords for mapped track
        for key, track in sorted(self.osm_tracks.items()):
            if key and "1-S" not in key:
                if track.lat and track.lon:
                    ax.plot(track.lon, track.lat, c=track.color, label=track.name)

        ax.set_aspect(asp, adjustable='datalim')

        ax.legend()
        tikzplotlib.save("network.tex")

        plt.show()

    def jensen_shannon_distance(self, s1, s2):
        """
        method to compute the Jenson-Shannon Distance
        between two probability distributions using imput signal s2 / s2
        """
        if not isinstance(s1, np.dtype):
            s1 = np.array(s1).reshape(-1, 1)

        if not isinstance(s2, np.dtype):
            s2 = np.array(s2).reshape(-1, 1)

        s1 = s1[s1 != np.inf]
        s2 = s2[s2 != np.inf]

        fig, ax = plt.subplots()
        p = sns.kdeplot(s1, ax=ax).get_lines()[0].get_data()
        plt.close()
        fig, ax = plt.subplots()
        q = sns.kdeplot(s2, ax=ax).get_lines()[0].get_data()
        plt.close()

        df_kde = pd.DataFrame({'kappa_ref': p[0],
                               'P_ref': p[1],
                               'kappa_osm': q[0],
                               'P_osm': q[1]
                               })

        p = p[1]
        q = q[1]

        m = (p + q) / 2

        divergence = (stats.entropy(p, m) + stats.entropy(q, m)) / 2
        distance = np.sqrt(divergence)

        return df_kde, distance

    def plot_track(self, names=None, matched=True, save_data=False):
        if names is None:
            names = ["a"]

        if isinstance(names, str):
            names = [names]

        if not isinstance(names, list):
            raise ValueError("Names has to be either a list of strings or a single string")

        for name in names:
            if name in self.reference_tracks.keys():
                ref_track = self.reference_tracks[name]

                ref_lon, ref_lat = ref_track.data["lon"], ref_track.data["lat"]

                asp = 1 / math.cos(math.radians((min(ref_lat) + max(ref_lat)) / 2))

                fig, ax = plt.subplots(1, 2)

                # Coord subplot
                ax[0].plot(ref_lon, ref_lat, label="Ref. coords")

                if matched and name in self.osm_tracks.keys():
                    ref_lon, ref_lat = self.osm_tracks[name].lon, self.osm_tracks[name].lat
                    ax[0].plot(ref_lon, ref_lat, label="OSM coords")
                    ax[0].plot(self.osm_tracks[name].lon_interp,
                               self.osm_tracks[name].lat_interp, label="OSM spline")

                ax[0].set_aspect(asp, adjustable='datalim')

                # Curvature subplot
                ref_s, ref_c = ref_track.data["s"], ref_track.data["curv_hor"]
                ax[1].plot(ref_s, ref_c, label="Ref. curvature")

                if matched and name in self.osm_tracks.keys():
                    osm_s, osm_c = self.osm_tracks[name].s, self.osm_tracks[name].c
                    ax[1].plot(osm_s, osm_c, label="OSM coords")
                    ax[1].plot(self.osm_tracks[name].s_interp,
                               self.osm_tracks[name].c_interp, label="OSM coords")

                fig.suptitle(name)
                plt.show()

                if save_data:
                    curv_ref = None
                    curv_osm = None
                    cdir = os.getcwd()
                    os.chdir('..')
                    os.chdir('..')

                    # reference track
                    try:
                        ref_track.data.to_csv(f'editorial/resultdata/results_track_{name}_ref.csv', index=False)
                        curv_ref = ref_track.data['curv_hor']
                        ref_track.data[['curv_hor']].describe().T.to_csv(f'editorial/resultdata/stats_track_{name}_ref.csv', index=False)
                    except FileNotFoundError:
                        print(FileNotFoundError)

                    # osm track
                    try:
                        df_temp = pd.DataFrame({'s': self.osm_tracks[name].s,
                                                'lat': self.osm_tracks[name].lat,
                                                'lon': self.osm_tracks[name].lon,
                                                'curv_hor': self.osm_tracks[name].c})
                        df_temp.to_csv(f'editorial/resultdata/results_track_{name}_osm.csv', index=False)
                        curv_osm = df_temp['curv_hor']
                        df_temp[['curv_hor']].describe().T.to_csv(f'editorial/resultdata/stats_track_{name}_osm.csv', index=False)
                    except FileNotFoundError:
                        print(FileNotFoundError)
                    except Exception as ex:
                        print(ex)

                    if curv_ref is not None and curv_osm is not None:
                        kde_df, jshdist = self.jensen_shannon_distance(curv_ref, curv_osm)
                        kde_df.to_csv(f'editorial/resultdata/results_track_{name}_kde.csv', index=False)

                        with open(f'editorial/resultdata/jensen_shannon_{name}.txt', 'w') as f:
                            f.write('{:.4f}'.format(jshdist))

                    os.chdir(cdir)

            else:
                raise UserWarning("%s not in reference tracks" % name)

        breakpoint()
