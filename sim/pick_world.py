import random

import jax
import jax.numpy as jp
import numpy as np

from evals.eval_pick import build
from perception.locate import CUBE_RGBA
from perception.pick_camera import PickCamera

# long enough for a cube left in the air to fall and land
SETTLE_STEPS = 25


class PickWorld:
    """The one simulated scene shared by the camera, the planner, and the arm.

    Build one per program: freeing an old world can break the next one's camera.
    """

    def __init__(self, seed: int = 0, object_id: str | None = None):
        self.config, self.env, self.act, self.reset, self.step = build()
        self.camera = PickCamera(self.env)
        self.cube_address = self.env._obj_qposadr
        self.key = jax.random.PRNGKey(seed)
        # one cube per episode, in the chosen color or a random one
        self.colors = random.Random(seed)
        self.choices = [object_id] if object_id else list(CUBE_RGBA)
        self.new_episode()

    def next_key(self):
        self.key, key = jax.random.split(self.key)
        return key

    def new_episode(self):
        # a new random cube and target, with the arm at home
        self.state = self.reset(self.next_key()[None])
        self.arm_at_home = True
        self.object_id = self.colors.choice(self.choices)
        self.camera.paint_cube(self.object_id)

    def home_arm(self):
        # puts the arm back at home for another try, leaving the cube where it ended up
        a = self.cube_address
        cube_pose = self.state.data.qpos[:, a:a + 7]
        state = self.reset(self.next_key()[None])
        qpos = state.data.qpos.at[:, a:a + 7].set(cube_pose)
        state = state.replace(data=state.data.replace(qpos=qpos))
        hold_still = jp.zeros((1, self.env.action_size))
        for _ in range(SETTLE_STEPS):
            state = self.step(state, hold_still)
        self.state = state
        self.arm_at_home = True

    def locate_cube(self):
        return self.camera.locate_cube(np.array(self.state.data.qpos[0]), self.object_id)

    def side_photo(self):
        rgb, _ = self.camera.take_photos(np.array(self.state.data.qpos[0]), "side")
        return rgb

    def observe(self):
        # same format as perception.scene.observe, but for this scene
        found = self.camera.see(np.array(self.state.data.qpos[0]))
        return {name: position.round(3).tolist() for name, position in found.items()}
