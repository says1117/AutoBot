from pathlib import Path

import jax.numpy as jp
from brax.io import model

from perception.pick_camera import CubeTracker, hide_cube_state
from planner.runtime import SetupError, log
from sim.pick_world import PickWorld
from skills.contract import SkillSpec
from skills.runner import AttemptResult

# tracked in git so a fresh clone can run; new training runs go to checkpoints/
POLICY_PATH = Path(__file__).parent.parent / "models" / "pick_policy_seed2"


class PolicyAttempter:
    def __init__(self, world: PickWorld, verifier=None):
        self.world = world
        self.verifier = verifier
        if not POLICY_PATH.exists():
            raise SetupError(f"policy file not found: {POLICY_PATH}. It should come with the repo.")
        self.params = model.load_params(str(POLICY_PATH))
        self.box = world.env.mj_model.body("box").id

    def attempt(self, spec: SkillSpec, args: dict) -> AttemptResult:
        world = self.world
        if args["object_id"] != world.object_id:
            return AttemptResult(outcome="failure", reason="that object is not in the scene")

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

        # the simulator's height is what a real robot wouldn't have, so a photo decides
        lifted = best_held >= hold_steps
        confirmed = self.verifier.confirm(world.side_photo(), args["object_id"]) if self.verifier else None
        if self.verifier:
            log.info("photo check: %s, height rule: %s", confirmed, lifted)
            if confirmed is None:
                log.warning("no answer from the photo check, so the height rule decides")
            elif confirmed != lifted:
                log.warning("the photo check and the height rule disagree")
        if confirmed is None:
            confirmed = lifted
        if confirmed:
            return AttemptResult(outcome="success")
        return AttemptResult(outcome="failure", reason="the check found the object was not held")
