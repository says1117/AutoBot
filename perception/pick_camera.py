from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from perception.camera import HEIGHT, WIDTH, capture
from perception.locate import COLOR_RANGES, CUBE_RGBA, back_project, find_pixels, is_corrupted

# looks back at the robot from in front of the table, so the arm does not block the cube
CAMERA_XML = '<camera name="overhead" pos="1.4 0 0.7" xyaxes="0 1 0 -0.7071 0 0.7071" fovy="60"/>'
# looks across the table from the side, so a photo shows how high the cube is off the floor
SIDE_CAMERA_XML = '<camera name="side" pos="0.55 -1.4 0.4" xyaxes="1 0 0 0 0 1" fovy="45"/>'
# measured distance from the cube's visible surface to its center, along the camera's view
SURFACE_OFFSET = 0.017
# once the gripper is this close to the estimate, the cube is assumed to move with it
GRASP_DISTANCE = 0.015
# under the floor, where the camera can't see it
HIDDEN = (0, 0, -5)
# a whole cube covers about 580 pixels, so fewer than this is not a cube
MIN_PIXELS = 50


class PickCamera:
    """A camera added to the same scene the policy was trained in."""

    def __init__(self, env):
        xml = Path(env.xml_path).read_text()
        xml = xml.replace("<worldbody>", "<worldbody>" + CAMERA_XML + SIDE_CAMERA_XML, 1)
        self.model = mujoco.MjModel.from_xml_string(xml, assets=env._model_assets)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, HEIGHT, WIDTH)
        self.cube_geoms = np.flatnonzero(self.model.geom_bodyid == self.model.body("box").id)
        # the red target marker is a guide for people, and would look like a red cube
        self.data.mocap_pos[:] = HIDDEN

    def paint_cube(self, object_id):
        # only changes how the cube looks; the physics and the policy never see color
        self.model.geom_rgba[self.cube_geoms] = CUBE_RGBA[object_id]

    def take_photos(self, qpos, camera="overhead"):
        # copy the episode's state in, then take the photos
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        rgb, depth = capture(self.model, self.data, self.renderer, camera)
        # static contains every color, so it would otherwise look like six cubes
        if is_corrupted(rgb):
            raise RuntimeError("the camera returned a corrupted image")
        return rgb, depth

    def find_in_photos(self, rgb, depth, object_id):
        xs, ys = find_pixels(rgb, object_id)
        if len(xs) < MIN_PIXELS:
            return None
        z = float(np.median(depth[ys, xs])) + SURFACE_OFFSET
        return back_project(self.model, self.data, "overhead", xs.mean(), ys.mean(), z, WIDTH, HEIGHT)

    def locate_cube(self, qpos, object_id):
        return self.find_in_photos(*self.take_photos(qpos), object_id)

    def see(self, qpos):
        # searches for every color, so the label comes from the photo and not the simulator
        rgb, depth = self.take_photos(qpos)
        found = {name: self.find_in_photos(rgb, depth, name) for name in COLOR_RANGES}
        return {name: position for name, position in found.items() if position is not None}


class CubeTracker:
    """Estimates where the cube is without asking the simulator."""

    def __init__(self, start_estimate):
        self.start = start_estimate
        self.grasped = jp.zeros(start_estimate.shape[0], dtype=bool)
        self.offset = jp.zeros_like(start_estimate)

    def update(self, gripper):
        # before the grasp the cube stays put; after it, the cube follows the gripper
        close = jp.linalg.norm(gripper - self.start, axis=1) < GRASP_DISTANCE
        just_grasped = ~self.grasped & close
        self.offset = jp.where(just_grasped[:, None], self.start - gripper, self.offset)
        self.grasped = self.grasped | just_grasped
        return jp.where(self.grasped[:, None], gripper + self.offset, self.start)


@jax.jit
def hide_cube_state(obs, estimate, target):
    """Replaces every exact cube value in the policy's input with the estimate."""
    gripper = obs[:, 31:34]
    obs = obs.at[:, 9:12].set(estimate)  # cube position
    obs = obs.at[:, 12:16].set(jp.array([1.0, 0, 0, 0]))  # cube rotation
    obs = obs.at[:, 25:31].set(0.0)  # cube velocity
    obs = obs.at[:, 40:46].set(jp.array([0.0, 1, 0, 0, 0, 1]))  # cube rotation matrix
    obs = obs.at[:, 46:49].set(estimate - gripper)  # cube relative to gripper
    obs = obs.at[:, 49:52].set(target - estimate)  # target relative to cube
    return obs.at[:, 52:58].set(0.0)  # rotation difference from target
