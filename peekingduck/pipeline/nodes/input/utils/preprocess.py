import cv2
import logging

logger = logging.getLogger(__name__)

def set_res(stream, desired_width, desired_height):
    '''
    Sets the resolution for the video frame
    '''
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    actual_width, actual_height = get_res(stream)
    if desired_width != actual_width:
        logger.warning("Unable to change width of video frame to %s, current width: %s!",
                            desired_width, actual_width)
    if desired_height != actual_height:
        logger.warning("Unable to change height of video frame to %s, current height: %s!",
                            desired_height, actual_height)

def get_res(stream):
    '''
    Gets the resolution for the video frame
    '''
    width = int(stream.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))

    return width, height

def mirror(frame):
    '''
    Mirrors a video frame.
    '''
    return cv2.flip(frame, 1)