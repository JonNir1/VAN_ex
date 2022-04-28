
from models.directions import Position
from models.frame import Frame
from models.match import FrameMatch, MutualMatch
from models.track import Track
from models.camera import Camera

# TODO: create new Tracks without actually instantiating a Track object (to remove the Track object type)


class Updater:

    def __init__(self, back_frame: Frame, front_frame: Frame, consensus_matches: list[MutualMatch]):
        self._back_frame = back_frame
        self._front_frame = front_frame
        self._consensus_matches = consensus_matches

    def update(self, non_consensus_front_matches: list[FrameMatch], fl_cam: Camera, fr_cam: Camera):
        self._update_existing_tracks()
        self._create_new_tracks(non_consensus_front_matches)
        self._update_front_cameras(fl_cam, fr_cam)

    def _update_existing_tracks(self):
        """
        For each of the Consensus Matches, finds the corresponding Track among $_back_frame's tracks,
            and update that track's _end_frame field.
        Also, this adds these tracks to the $_front_frame's _tracks field.
        """
        for i, con in enumerate(self._consensus_matches):
            back_match = con.get_frame_match(Position.BACK)
            front_match = con.get_frame_match(Position.FRONT)
            trk_id = self._back_frame.find_track_id_for_match(back_match)
            assert trk_id is not None, \
                f"Consensus Match #{i} between {str(self._back_frame)} and {str(self._front_frame)}" + \
                "has no associated Track"
            self._front_frame.add_track(trk_id, front_match)

    def _create_new_tracks(self, non_consensus_front_matches: list[FrameMatch]):
        """
        Iterates over the FrameMatches associated with the $_front_frame and creates a new Track object
            for new matches, or finds the existing Track object for the Consensus Matches
        """
        front_frame_idx = self._front_frame.id
        for m in non_consensus_front_matches:
            tr = Track(start_frame_idx=front_frame_idx)
            self._front_frame.add_track(tr.get_id(), m)

    def _update_front_cameras(self, left_cam: Camera, right_cam: Camera):
        self._front_frame.left_camera = left_cam
        self._front_frame.right_camera = right_cam
