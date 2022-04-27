import cv2


class Matcher:

    def __init__(self, matcher_name: str, cross_check=True):
        self.cross_check = cross_check
        self.type = matcher_name.upper()
        self._matcher = self.__create_matcher()

    def match(self, queryDesc, trainDesc):
        return self._matcher.match(queryDesc, trainDesc)

    def __create_matcher(self):
        # create a cv2.matcher object
        if self.type == "BF":
            return cv2.BFMatcher(norm=cv2.NORM_L2, cross_check=self.cross_check)
        if self.type == "FLANN":
            if self.cross_check:
                # TODO
                raise NotImplementedError(f"We currently do not support cross-check with {str(self)}")
            return cv2.FlannBasedMatcher(indexParams=dict(algorithm=0, trees=5), searchParams=dict(checks=50))
        raise NotImplementedError(f"We currently do not support the \"{self.type}\" Matcher")

    def __str__(self):
        return f"{self.type}Matcher"
