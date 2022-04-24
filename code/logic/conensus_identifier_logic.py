
import config as c
import utils as u
from models.frame import Frame
from models.camera import Camera


class ConsensusIdentifierLogic:
    """
    For two consecutive Frames, a consensus is a match that corresponds to all four images
    """

    def __init__(self, back_frame: Frame, front_frame: Frame):
        self._back_frame = back_frame
        self._front_frame = front_frame

    def identify_tracks_between_frames(self, **kwargs):
        # TODO
        back_tracks = self._back_frame.get_tracks()


        matcher_name = kwargs.get("matcher_name", c.DEFAULT_MATCHER_NAME)
        kps_back_left, desc_back_left, kps_back_right, _, back_matches = self._back_frame.detect_and_match(matcher_name=matcher_name)
        kps_front_left, desc_front_left, kps_front_right, _, front_matches = self._front_frame.detect_and_match(matcher_name=matcher_name)
        matcher = u.create_matcher(matcher_name)
        tracking_matches = matcher.match(desc_back_left, desc_front_left)


