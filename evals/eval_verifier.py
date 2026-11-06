import argparse
import json
from datetime import datetime

import jax
import jax.numpy as jp
import numpy as np
from brax.io import model

from evals.eval_pick import RESULTS, build
from perception.locate import CUBE_RGBA
from perception.pick_camera import CubeTracker, PickCamera, hide_cube_state
from perception.verify import MODEL, VLMVerifier
from skills.policy_attempter import POLICY_PATH
from skills.registry import load_registry


def final_poses(camera, parts, n_episodes, seed):
    """Runs real episodes and returns where each one ended and how high the cube finished."""
    config, env, act, reset, step = parts
    params = model.load_params(str(POLICY_PATH))
    address = env._obj_qposadr

    key = jax.random.PRNGKey(seed)
    state = reset(jax.random.split(key, n_episodes))
    start = np.array(state.data.qpos)
    target = state.info["target_pos"]
    found = [camera.locate_cube(start[i], "green_cube") for i in range(n_episodes)]
    tracker = CubeTracker(jp.array(np.array(found)))

    last = start.copy()
    running = np.ones(n_episodes, dtype=bool)
    for _ in range(config.episode_length):
        key, step_key = jax.random.split(key)
        obs = hide_cube_state(state.obs, tracker.update(state.obs[:, 31:34]), target)
        state = step(state, act(params, obs, step_key))
        running &= np.array(state.done) == 0
        last[running] = np.array(state.data.qpos)[running]
    return last, last[:, address + 2] - start[:, address + 2]


def evaluate(n_episodes, seed, per_class, wrong_color):
    parts = build()
    camera = PickCamera(parts[1])
    camera.paint_cube("green_cube")
    last, lift = final_poses(camera, parts, n_episodes, seed)
    held = lift >= load_registry()["pick"].outcomes.success.threshold

    # the same number of good and bad endings, since bad ones are rare
    rng = np.random.default_rng(0)
    good = rng.permutation(np.flatnonzero(held))[:per_class]
    bad = rng.permutation(np.flatnonzero(~held))[:per_class]
    colors = list(CUBE_RGBA)
    questions = []
    for n, episode in enumerate([*good, *bad]):
        paint = colors[n % len(colors)]
        questions.append({"episode": int(episode), "paint": paint, "asked": paint, "kind": "held" if held[episode] else "failed"})
    # a held cube, but the question names a color that isn't there, so the right answer is no
    for question in questions[:wrong_color]:
        asked = colors[(colors.index(question["paint"]) + 3) % len(colors)]
        questions.append({**question, "asked": asked, "kind": "wrong_color"})

    verifier = VLMVerifier()
    for question in questions:
        camera.paint_cube(question["paint"])
        rgb, _ = camera.take_photos(last[question["episode"]], "side")
        question["answer"] = verifier.confirm(rgb, question["asked"])
        question["lift_m"] = round(float(lift[question["episode"]]), 3)
        print(question, flush=True)

    def said_yes(kind):
        group = [q for q in questions if q["kind"] == kind and q["answer"] is not None]
        return {"yes": sum(q["answer"] for q in group), "of": len(group)}

    scores = {kind: said_yes(kind) for kind in ("held", "failed", "wrong_color")}
    answered = sum(score["of"] for score in scores.values())
    right = scores["held"]["yes"] + answered - scores["held"]["of"] - scores["failed"]["yes"] - scores["wrong_color"]["yes"]
    return {
        "model": MODEL,
        "camera": "side",
        "seed": seed,
        "questions": len(questions),
        "answered": answered,
        "right": right,
        "said_yes": scores,
        "details": questions,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--per-class", type=int, default=30)
    parser.add_argument("--wrong-color", type=int, default=10)
    args = parser.parse_args()

    result = evaluate(args.episodes, args.seed, args.per_class, args.wrong_color)
    print({key: value for key, value in result.items() if key != "details"}, flush=True)

    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / f"verifier_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.write_text(json.dumps(result, indent=2))
    print(f"saved {output}", flush=True)
