import math

import cv2
import numpy as np

# (lowest HSV, highest HSV). OpenCV hue runs 0-179.
# Red has two ranges because its hue wraps around the end of the scale.
COLOR_RANGES = {
    "red_cube": [((0, 120, 60), (10, 255, 255)), ((170, 120, 60), (179, 255, 255))],
    "orange_cube": [((11, 120, 60), (24, 255, 255))],
    "yellow_cube": [((25, 120, 60), (35, 255, 255))],
    "green_cube": [((45, 120, 60), (75, 255, 255))],
    # the training floor is blue too: hue 110 clears the floor, and saturation 200 clears
    # the cube's reflection in it
    "blue_cube": [((110, 200, 60), (130, 255, 255))],
    "purple_cube": [((135, 120, 60), (155, 255, 255))],
}

# the paint for each cube, as red, green, blue, alpha from 0 to 1
CUBE_RGBA = {
    "red_cube": (1, 0, 0, 1),
    # darker than pure orange, because bright light turns pure orange's top face yellow
    "orange_cube": (0.7, 0.3, 0, 1),
    "yellow_cube": (1, 1, 0, 1),
    "green_cube": (0, 1, 0, 1),
    "blue_cube": (0, 0, 1, 1),
    "purple_cube": (0.6, 0, 1, 1),
}

# The camera sees the cube's surface, but we want its center.
SURFACE_OFFSET = 0.025 * math.cos(math.radians(45))

def is_corrupted(rgb):
    # static jumps by about 70 between neighboring pixels, while a real photo barely changes
    return np.abs(np.diff(rgb.astype(np.int16), axis=1)).mean() > 30


def find_pixels(rgb, object_id):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    # start with an all-black mask, same height and width as the image
    mask = np.zeros(rgb.shape[:2], np.uint8)
    for low, high in COLOR_RANGES[object_id]:
        # mark pixels in this color range white; |= keeps earlier ranges too
        mask |= cv2.inRange(hsv, np.array(low), np.array(high))
    # returns coordinate of every white pixel (rows (y) first, then columns (x))
    ys, xs = np.nonzero(mask)
    return xs, ys

def back_project(model, data, camera, u, v, depth, width, height):
    # 1. get the camera's zoom as a focal length, in pixels
    cam = model.camera(camera).id
    fovy = math.radians(model.cam_fovy[cam])
    f = (height / 2) / math.tan(fovy / 2)
    cx, cy = width / 2, height / 2

    # 2. pixel + depth -> point relative to the camera, in meters
    x = (u - cx) * depth / f
    y = -(v - cy) * depth / f
    z = -depth
    p_cam = np.array([x, y, z])

    # 3. relative to the camera -> position in the world
    rotation = data.cam_xmat[cam].reshape(3, 3)
    return rotation @ p_cam + data.cam_xpos[cam]


def locate(model, data, rgb, depth_image, object_id, camera="overhead"):
    xs, ys = find_pixels(rgb, object_id)
    # if a cube is hidden it returns none
    if len(xs) == 0:
        return None
    u, v = xs.mean(), ys.mean()
    z = np.median(depth_image[ys, xs]) + SURFACE_OFFSET
    height, width = depth_image.shape
    return back_project(model, data, camera, u, v, z, width, height)