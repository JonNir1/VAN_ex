import time
import numpy as np
from itertools import count

import config as c
from models.matcher import Matcher
from models.Frame2 import Frame2


class FrameProcessor2:

    def __init__(self, matcher: Matcher = None, verbose=False):
        if matcher is None:
            self._matcher = c.MATCHER
        else:
            self._matcher = matcher
        self._verbose = verbose
        self._track_count = count(0)

    def _process_frame_pair(self, back_frame: Frame2, front_frame: Frame2):
        cv2_matches = self._matcher.match(back_frame.descriptors, front_frame.descriptors)
        back_features = np.array([back_frame.features[m.trainIdx] for m in cv2_matches])
        front_features = np.array([front_frame.features[m.trainIdx] for m in cv2_matches])
        return


