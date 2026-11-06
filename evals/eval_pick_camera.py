import argparse
import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np
from brax.io import model

from evals.eval_pick import ENV_NAME, RESULTS, build
from perception.locate import CUBE_RGBA
from perception.pick_camera import CubeTracker, PickCamera, hide_cube_state
from skills.policy_attempter import POLICY_PATH
from skills.registry import load_registry


def evaluate(policy_path, n_episodes, seed, object_id):
    config, env, act, reset, step = build()
    camera = PickCamera(env)
    camera.paint_cube(object_id)
    rule = load_registry()["pick"].outcomes.success
    hold_steps = round(rule.hold_s / config.ctrl_dt)
    box = env.mj_model.body("box").id
    params = model.load_params(policy_path)

    key = jax.random.PRNGKey(seed)
    state = reset(jax.random.split(key, n_episodes))
    cube_address = env._obj_qposadr
    true_start = np.array(state.data.qpos[:, cube_address:cube_address + 3])
    start_height = jp.array(true_start[:, 2])
    target = state.info["target_pos"]

    # one photo per episode, taken before the arm moves
    qpos = np.array(state.data.qpos)
    found = [camera.locate_cube(qpos[i], object_id) for i in range(n_episodes)]
    misses = sum(position is None for position in found)
    estimates = np.array([np.zeros(3) if p is None else p for p in found])
    seen = np.array([p is not None for p in found])
    camera_error = np.linalg.norm(estimates[seen] - true_start[seen], axis=1)

    tracker = CubeTracker(jp.array(estimates))
    held = jp.zeros(n_episodes, dtype=jp.int32)
    best_held = jp.zeros(n_episodes, dtype=jp.int32)
    active = jp.ones(n_episodes, dtype=bool)

    for _ in range(config.episode_length):
        key, step_key = jax.random.split(key)
        estimate = tracker.update(state.obs[:, 31:34])
        obs = hide_cube_state(state.obs, estimate, target)
        state = step(state, act(params, obs, step_key))
        active = active & (state.done == 0)

        lift = state.data.xpos[:, box, 2] - start_height
        lifted = active & (lift >= rule.threshold)
        held = jp.where(lifted, held + 1, 0)
        best_held = jp.maximum(best_held, held)

    successes = int((best_held >= hold_steps).sum())
    target_error = jp.linalg.norm(state.data.xpos[:, box, :] - target, axis=1)
    return {
        "policy": Path(policy_path).name,
        "attempter": "ppo_camera",
        "object": object_id,
        "env": ENV_NAME,
        "episodes": n_episodes,
        "seed": seed,
        "lift_threshold_m": rule.threshold,
        "hold_s": rule.hold_s,
        "successes": successes,
        "success_rate": successes / n_episodes,
        "camera_misses": misses,
        "mean_camera_error_mm": round(float(camera_error.mean() * 1000), 1),
        "max_camera_error_mm": round(float(camera_error.max() * 1000), 1),
        "mean_final_target_error_m": round(float(target_error.mean()), 3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", nargs="?", default=str(POLICY_PATH))
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--object", choices=list(CUBE_RGBA), default="green_cube")
    args = parser.parse_args()

    result = evaluate(args.policy, args.episodes, args.seed, args.object)
    print(result, flush=True)

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = RESULTS / f"pick_ppo_camera_{args.object}_{stamp}.json"
    output.write_text(json.dumps(result, indent=2))
    print(f"saved {output}", flush=True)
