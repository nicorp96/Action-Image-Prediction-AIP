import cv2


def canny(image, use_cuda=False, t_lower=100, t_upper=200, aperture_size=5):
    edge = cv2.Canny(image, t_lower, t_upper, apertureSize=aperture_size)
    return edge
