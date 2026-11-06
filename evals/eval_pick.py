import argparse
import json
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jp
from brax.io import model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry
from mujoco_playground.config import manipulation_params

from skills.registry import load_registry

ENV_NAME = "PandaPickCube"
RESULTS = Path(__file__).parent / "results"


def build():
    # the same task and network shape the policy was trained with
    config = registry.get_default_config(ENV_NAME)
    env = registry.load(ENV_NAME, config=config)
    settings = manipulation_params.brax_ppo_config(ENV_NAME).to_dict()["network_factory"]
    networks = ppo_networks.make_ppo_networks(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
        **settings,
    )
    make_policy = ppo_networks.make_inference_fn(networks)

    def act(params, obs, key):
        return make_policy(params, deterministic=True)(obs, key)[0]

    return config, env, jax.jit(act), jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))


def evaluate(policy_path, parts, n_episodes, seed):
    config, env, act, reset, step = parts
    success_rule = load_registry()["pick"].outcomes.success
    hold_steps = round(success_rule.hold_s / config.ctrl_dt)
    box = env.mj_model.body("box").id
    params = model.load_params(policy_path)

    key = jax.random.PRNGKey(seed)
    state = reset(jax.random.split(key, n_episodes))
    # xpos is all zeros until the first step, so read the height from qpos
    start_height = state.data.qpos[:, env._obj_qposadr + 2]

    # counts how many steps in a row each cube has stayed lifted
    held = jp.zeros(n_episodes, dtype=jp.int32)
    best_held = jp.zeros(n_episodes, dtype=jp.int32)
    best_lift = jp.zeros(n_episodes)
    active = jp.ones(n_episodes, dtype=bool)

    for _ in range(config.episode_length):
        key, step_key = jax.random.split(key)
        state = step(state, act(params, state.obs, step_key))
        active = active & (state.done == 0)

        lift = state.data.xpos[:, box, 2] - start_height
        lifted = active & (lift >= success_rule.threshold)
        held = jp.where(lifted, held + 1, 0)
        best_held = jp.maximum(best_held, held)
        best_lift = jp.maximum(best_lift, jp.where(active, lift, 0.0))

    box_position = state.data.xpos[:, box, :]
    target_error = jp.linalg.norm(box_position - state.info["target_pos"], axis=1)
    successes = int((best_held >= hold_steps).sum())

    return {
        "policy": Path(policy_path).name,
        "attempter": "ppo",
        "env": ENV_NAME,
        "episodes": n_episodes,
        "seed": seed,
        "lift_threshold_m": success_rule.threshold,
        "hold_s": success_rule.hold_s,
        "successes": successes,
        "success_rate": successes / n_episodes,
        "mean_best_lift_m": round(float(best_lift.mean()), 3),
        "mean_final_target_error_m": round(float(target_error.mean()), 3),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("policies", nargs="+")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    parts = build()
    results = [evaluate(path, parts, args.episodes, args.seed) for path in args.policies]
    for result in results:
        print(result, flush=True)

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = RESULTS / f"pick_ppo_{stamp}.json"
    output.write_text(json.dumps(results, indent=2))
    print(f"saved {output}", flush=True)
