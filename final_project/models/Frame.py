

import final_project.utils as u


class Frame:

    _MaxIndex = 3449

    def __init__(self, idx: int):
        if idx < 0 or idx > Frame._MaxIndex:
            raise IndexError(f"Frame index must be between 0 and {Frame._MaxIndex}, not {idx}")
        self.idx = idx
        return None

    def _detect_and_match(self):
        img_l, img_r = u.read_images(self.idx)
        return None

