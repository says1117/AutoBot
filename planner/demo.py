from planner.planner import run_instruction
from skills.policy_attempter import PolicyAttempter

if __name__ == "__main__":
    print(run_instruction("Pick up the green cube.", PolicyAttempter()))
