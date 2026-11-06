from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.io import model

from evals.eval_pick import build
from perception.pick_camera import CubeTracker, PickCamera, hide_cube_state
from skills.contract import SkillSpec
from skills.runner import AttemptResult

POLICY_PATH = Path(__file__).parent.parent / "checkpoints" / "pick_policy_seed2"
# the training scene has one cube, and it is green
TRAINED_OBJECT = "green_cube"


class PolicyAttempter:
    def __init__(self, seed: int = 0):
        # the slow setup happens once, not on every attempt
        self.config, self.env, self.act, self.reset, self.step = build()
        self.params = model.load_params(str(POLICY_PATH))
        self.box = self.env.mj_model.body("box").id
        self.camera = PickCamera(self.env)
        self.key = jax.random.PRNGKey(seed)

    def attempt(self, spec: SkillSpec, args: dict) -> AttemptResult:
        if args["object_id"] != TRAINED_OBJECT:
            return AttemptResult(outcome="failure", reason="no trained policy for this object")

        # the success rule comes from skill.yaml, the same one eval_pick.py uses
        rule = spec.outcomes.success
        hold_steps = round(rule.hold_s / self.config.ctrl_dt)

        # new episode; [None] makes a batch of one because reset and step are batched
        self.key, reset_key = jax.random.split(self.key)
        state = self.reset(reset_key[None])
        # xpos is all zeros until the first step, so read the height from qpos
        start_height = float(state.data.qpos[0, self.env._obj_qposadr + 2])

        # the policy works from one camera photo, not the simulator's exact cube position
        estimate = self.camera.locate_cube(np.array(state.data.qpos[0]), np.array(state.data.mocap_pos[0]))
        if estimate is None:
            return AttemptResult(outcome="failure", reason="camera could not see the object")
        tracker = CubeTracker(jp.array(estimate)[None])
        target = state.info["target_pos"]

        held = 0
        best_held = 0
        for _ in range(self.config.episode_length):
            self.key, step_key = jax.random.split(self.key)
            obs = hide_cube_state(state.obs, tracker.update(state.obs[:, 31:34]), target)
            action = self.act(self.params, obs, step_key)
            state = self.step(state, action)

            # the episode ended early, such as the cube leaving the work area
            if float(state.done[0]) != 0:
                break

            # count how many steps in a row the cube has stayed lifted
            lift = float(state.data.xpos[0, self.box, 2]) - start_height
            held = held + 1 if lift >= rule.threshold else 0
            best_held = max(best_held, held)

        if best_held >= hold_steps:
            return AttemptResult(outcome="success")
        return AttemptResult(outcome="failure", reason="object was not held at lift height")
