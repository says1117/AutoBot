import cv2
import mujoco
import numpy as np
import pytest

from perception.locate import (
    COLOR_RANGES,
    CUBE_RGBA,
    SURFACE_OFFSET,
    back_project,
    find_pixels,
    is_corrupted,
    locate,
)

WIDTH, HEIGHT = 640, 480
# one meter up, looking straight down. fovy=90 makes the focal length exactly HEIGHT / 2
STRAIGHT_DOWN = '<camera name="overhead" pos="0 0 1" xyaxes="1 0 0 0 1 0" fovy="90"/>'
# tilted like the real scene camera, so the rotation step is really tested
TILTED = '<camera name="overhead" pos="0.5 -0.7 0.8" xyaxes="1 0 0 0 0.7 0.7" fovy="45"/>'


def camera_scene(camera_xml):
    model = mujoco.MjModel.from_xml_string(f"<mujoco><worldbody>{camera_xml}</worldbody></mujoco>")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def image_with_square(color, left=100, top=200, size=20):
    rgb = np.full((HEIGHT, WIDTH, 3), 128, np.uint8)
    rgb[top:top + size, left:left + size] = color
    return rgb


def paint(object_id):
    return tuple(round(channel * 255) for channel in CUBE_RGBA[object_id][:3])


@pytest.mark.parametrize("object_id", CUBE_RGBA)
def test_finds_each_color_where_it_is(object_id):
    xs, ys = find_pixels(image_with_square(paint(object_id)), object_id)
    assert len(xs) == 20 * 20
    assert (xs.mean(), ys.mean()) == (109.5, 209.5)


@pytest.mark.parametrize("object_id", CUBE_RGBA)
def test_each_paint_matches_only_its_own_color(object_id):
    image = image_with_square(paint(object_id))
    matches = [name for name in COLOR_RANGES if len(find_pixels(image, name)[0]) > 0]
    assert matches == [object_id]


def test_training_floor_is_not_mistaken_for_a_blue_cube():
    # the brightest, most saturated floor color measured in the training scene
    floor = cv2.cvtColor(np.array([[[107, 174, 255]]], np.uint8), cv2.COLOR_HSV2RGB)[0, 0]
    assert len(find_pixels(image_with_square(tuple(int(c) for c in floor)), "blue_cube")[0]) == 0


def test_red_is_found_at_the_far_end_of_the_hue_scale():
    # this red sits near hue 175, which only the second red range covers
    xs, _ = find_pixels(image_with_square((255, 0, 40)), "red_cube")
    assert len(xs) == 20 * 20


def test_gray_floor_matches_no_color():
    gray = np.full((HEIGHT, WIDTH, 3), 128, np.uint8)
    for object_id in COLOR_RANGES:
        assert len(find_pixels(gray, object_id)[0]) == 0


def test_static_counts_as_corrupted():
    static = np.random.default_rng(0).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)
    assert is_corrupted(static)


def test_ordinary_image_is_not_corrupted():
    assert not is_corrupted(image_with_square((0, 255, 0)))


def test_center_pixel_lands_straight_below_the_camera():
    model, data = camera_scene(STRAIGHT_DOWN)
    point = back_project(model, data, "overhead", WIDTH / 2, HEIGHT / 2, 1.0, WIDTH, HEIGHT)
    assert point == pytest.approx([0, 0, 0])


def test_off_center_pixel_matches_the_math_done_by_hand():
    model, data = camera_scene(STRAIGHT_DOWN)
    # f = 240, so 120 pixels right at 1 m is 0.5 m, and 60 pixels up is 0.25 m
    point = back_project(model, data, "overhead", 320 + 120, 240 - 60, 1.0, WIDTH, HEIGHT)
    assert point == pytest.approx([0.5, 0.25, 0])


def test_tilted_camera_gets_back_the_point_it_was_shown():
    model, data = camera_scene(TILTED)
    point = np.array([0.42, 0.13, 0.05])

    # go forward from the world to a pixel, with the opposite math to back_project
    rotation = data.cam_xmat[0].reshape(3, 3)
    x, y, z = rotation.T @ (point - data.cam_xpos[0])
    depth = -z
    f = (HEIGHT / 2) / np.tan(np.radians(45) / 2)
    u = WIDTH / 2 + f * x / depth
    v = HEIGHT / 2 - f * y / depth

    assert back_project(model, data, "overhead", u, v, depth, WIDTH, HEIGHT) == pytest.approx(point)


def test_locate_moves_from_the_surface_to_the_cube_center():
    model, data = camera_scene(STRAIGHT_DOWN)
    rgb = image_with_square((0, 255, 0), left=310, top=230)
    depth = np.ones((HEIGHT, WIDTH))
    position = locate(model, data, rgb, depth, "green_cube")
    assert position[2] == pytest.approx(-SURFACE_OFFSET)


def test_locate_ignores_a_stray_depth_value():
    model, data = camera_scene(STRAIGHT_DOWN)
    rgb = image_with_square((0, 255, 0), left=310, top=230)
    depth = np.ones((HEIGHT, WIDTH))
    depth[230, 310] = 50.0
    position = locate(model, data, rgb, depth, "green_cube")
    assert position[2] == pytest.approx(-SURFACE_OFFSET)


def test_locate_returns_none_for_a_hidden_cube():
    model, data = camera_scene(STRAIGHT_DOWN)
    rgb = image_with_square((0, 255, 0))
    assert locate(model, data, rgb, np.ones((HEIGHT, WIDTH)), "red_cube") is None
