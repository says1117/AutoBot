import sys

from perception.verify import VLMVerifier
from planner.planner import run_instruction
from sim.pick_world import PickWorld
from skills.policy_attempter import PolicyAttempter

if __name__ == "__main__":
    instruction = " ".join(sys.argv[1:]) or "Pick up the cube."
    world = PickWorld()
    print("camera sees:", world.observe())
    print(run_instruction(instruction, PolicyAttempter(world, VLMVerifier()), observe_fn=world.observe))
