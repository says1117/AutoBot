import jax
import numpy as np

from evals.eval_pick import build
from perception.pick_camera import PickCamera


class PickWorld:
    """The one simulated scene shared by the camera, the planner, and the arm."""

    def __init__(self, seed: int = 0):
        self.config, self.env, self.act, self.reset, self.step = build()
        self.camera = PickCamera(self.env)
        self.cube_address = self.env._obj_qposadr
        self.key = jax.random.PRNGKey(seed)
        self.new_episode()

    def next_key(self):
        self.key, key = jax.random.split(self.key)
        return key

    def new_episode(self):
        # a new random cube and target, with the arm at home
        self.state = self.reset(self.next_key()[None])
        self.arm_at_home = True

    def home_arm(self):
        # puts the arm back at home for another try, leaving the cube where it ended up
        a = self.cube_address
        cube_pose = self.state.data.qpos[:, a:a + 7]
        state = self.reset(self.next_key()[None])
        qpos = state.data.qpos.at[:, a:a + 7].set(cube_pose)
        self.state = state.replace(data=state.data.replace(qpos=qpos))
        self.arm_at_home = True

    def locate_cube(self):
        qpos = np.array(self.state.data.qpos[0])
        mocap = np.array(self.state.data.mocap_pos[0])
        return self.camera.locate_cube(qpos, mocap)

    def observe(self):
        # same format as perception.scene.observe, but for this scene
        position = self.locate_cube()
        if position is None:
            return {}
        return {"green_cube": position.round(3).tolist()}
