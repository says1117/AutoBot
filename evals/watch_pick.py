import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from planner.planner import run_instruction
from sim.pick_world import PickWorld
from skills.policy_attempter import PolicyAttempter


def show(viewer, model, data, qpos, mocap_pos):
    with viewer.lock():
        data.qpos[:] = qpos
        data.mocap_pos[:] = mocap_pos
        mujoco.mj_forward(model, data)
    viewer.sync()


def main(instruction, episodes):
    world = PickWorld()
    attempter = PolicyAttempter(world)

    # the arm runs on the GPU, so save every step and replay it in the window
    frames = []
    run_step = world.step

    def recording_step(state, action):
        state = run_step(state, action)
        frames.append((np.array(state.data.qpos[0]), np.array(state.data.mocap_pos[0])))
        return state

    world.step = recording_step

    # the window gets its own data so it never fights with the camera's
    model = world.camera.model
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.5, 0.0, 0.2]
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.8, 140, -20

        finished = 0
        while viewer.is_running() and (episodes == 0 or finished < episodes):
            world.new_episode()
            frames.clear()
            show(viewer, model, data, np.array(world.state.data.qpos[0]), np.array(world.state.data.mocap_pos[0]))

            print("camera sees:", world.observe(), flush=True)
            print("thinking and running the arm...", flush=True)
            print(run_instruction(instruction, attempter, observe_fn=world.observe), flush=True)

            start = time.time()
            for qpos, mocap_pos in frames:
                if not viewer.is_running():
                    break
                show(viewer, model, data, qpos, mocap_pos)
                time.sleep(world.config.ctrl_dt)
            print(f"replayed {len(frames)} steps in {time.time() - start:.1f}s", flush=True)

            time.sleep(2)
            finished += 1

        # let the window finish closing, or it crashes while Python shuts down
        viewer.close()
        while viewer.is_running():
            time.sleep(0.05)
    time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("instruction", nargs="?", default="Pick up the green cube.")
    parser.add_argument("--episodes", type=int, default=0, help="0 keeps going until the window is closed")
    args = parser.parse_args()
    main(args.instruction, args.episodes)
