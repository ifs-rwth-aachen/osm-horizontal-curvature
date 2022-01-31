from curvy import Curvy

from dataloader.dataset import Dataset
from utils.utils import OSMTrack


def run():
    coords = {"Mannheim": (8.03, 49.28, 8.89, 49.72)}

    curvy = Curvy(*coords["Mannheim"], desired_railway_types=["tram", "light_rail"])
    curvy.download_track_data()

    ref_dataset = Dataset("data/mannheim.pkl")
    for ref_track in ref_dataset.reference_tracks.values():
        ways = curvy.search_curvy_result(ref_track.way_ids, railway_type="tram")
        osm_track = OSMTrack(ways, identifier=ref_track.identifier, name=ref_track.name, color=ref_track.color)

        if ref_track.reverse:
            osm_track.reverse_track()
        if ref_track.flip_curvature:
            osm_track.flip_curvature()

        if ref_track.identifier in ref_dataset.mappings.keys():
            osm_track.offset = ref_dataset.mappings[ref_track.identifier]["offset"]

        if ref_track.name:
            ref_dataset.osm_tracks[ref_track.name] = osm_track

        ref_dataset.osm_tracks[ref_track.identifier] = osm_track

    ref_dataset.plot_network()
    ref_dataset.plot_track([mapping["name"] for mapping in ref_dataset.mappings.values()])
    # breakpoint()


if __name__ == "__main__":
    run()
