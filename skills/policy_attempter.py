from pathlib import Path

import jax.numpy as jp
from brax.io import model

from perception.pick_camera import CubeTracker, hide_cube_state
from sim.pick_world import PickWorld
from skills.contract import SkillSpec
from skills.runner import AttemptResult

POLICY_PATH = Path(__file__).parent.parent / "checkpoints" / "pick_policy_seed2"
# the training scene has one cube, and it is green
TRAINED_OBJECT = "green_cube"


class PolicyAttempter:
    def __init__(self, world: PickWorld):
        self.world = world
        self.params = model.load_params(str(POLICY_PATH))
        self.box = world.env.mj_model.body("box").id

    def attempt(self, spec: SkillSpec, args: dict) -> AttemptResult:
        if args["object_id"] != TRAINED_OBJECT:
            return AttemptResult(outcome="failure", reason="no trained policy for this object")

        world = self.world
        # a retry starts with the arm at home and a new photo of wherever the cube is now
        if not world.arm_at_home:
            world.home_arm()
        world.arm_at_home = False

        # the success rule comes from skill.yaml, the same one eval_pick.py uses
        rule = spec.outcomes.success
        hold_steps = round(rule.hold_s / world.config.ctrl_dt)

        # the policy works from one camera photo, not the simulator's exact cube position
        estimate = world.locate_cube()
        if estimate is None:
            return AttemptResult(outcome="failure", reason="camera could not see the object")
        tracker = CubeTracker(jp.array(estimate)[None])
        target = world.state.info["target_pos"]
        # xpos is all zeros until the first step, so read the height from qpos
        start_height = float(world.state.data.qpos[0, world.cube_address + 2])

        held = 0
        best_held = 0
        for _ in range(world.config.episode_length):
            state = world.state
            obs = hide_cube_state(state.obs, tracker.update(state.obs[:, 31:34]), target)
            action = world.act(self.params, obs, world.next_key())
            world.state = world.step(state, action)

            # the episode ended early, such as the cube leaving the work area
            if float(world.state.done[0]) != 0:
                break

            # count how many steps in a row the cube has stayed lifted
            lift = float(world.state.data.xpos[0, self.box, 2]) - start_height
            held = held + 1 if lift >= rule.threshold else 0
            best_held = max(best_held, held)

        if best_held >= hold_steps:
            return AttemptResult(outcome="success")
        return AttemptResult(outcome="failure", reason="object was not held at lift height")
