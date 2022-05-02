import cv2


def create_detector(detector_name: str):
    # create a cv2 feature detector
    detector_name = detector_name.upper()
    if detector_name == "ORB":
        return cv2.ORB_create()
    if detector_name == "SIFT":
        return cv2.SIFT_create()
    raise NotImplementedError(f"We currently do not support the {detector_name} detector")

