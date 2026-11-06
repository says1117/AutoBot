import sys

from planner.planner import run_instruction
from sim.pick_world import PickWorld
from skills.policy_attempter import PolicyAttempter

if __name__ == "__main__":
    instruction = " ".join(sys.argv[1:]) or "Pick up the green cube."
    world = PickWorld()
    print("camera sees:", world.observe())
    print(run_instruction(instruction, PolicyAttempter(world), observe_fn=world.observe))
