import sys

from perception.verify import VLMVerifier
from planner.planner import run_instruction
from planner.runtime import SetupError, log, setup_logging
from sim.pick_world import PickWorld
from skills.policy_attempter import PolicyAttempter


def main(instruction):
    log_path = setup_logging()
    # built first, so a missing key shows up before the slow startup
    verifier = VLMVerifier()
    world = PickWorld()
    print(run_instruction(instruction, PolicyAttempter(world, verifier), observe_fn=world.observe))
    log.info("log saved to %s", log_path)


if __name__ == "__main__":
    try:
        main(" ".join(sys.argv[1:]) or "Pick up the cube.")
    except SetupError as error:
        sys.exit(f"setup error: {error}")
