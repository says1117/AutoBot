import argparse
import functools
import time
from pathlib import Path

from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import manipulation_params

ENV_NAME = "PandaPickCube"
CHECKPOINTS = Path(__file__).parent.parent / "checkpoints"


def main(seed):
    # the simulated arm, cube, and reward
    config = registry.get_default_config(ENV_NAME)
    env = registry.load(ENV_NAME, config=config)

    # PPO settings; network sizes are passed separately from the rest
    params = manipulation_params.brax_ppo_config(ENV_NAME).to_dict()
    network_settings = params.pop("network_factory")
    params.update(num_evals=11, num_resets_per_eval=0, seed=seed)

    start = time.time()

    # brax calls this after each evaluation
    def progress(step, metrics):
        reward = float(metrics["eval/episode_reward"])
        box_target = float(metrics["eval/episode_box_target"])
        print(f"step {step:,} | reward {reward:.1f} | box_target {box_target:.1f} | {time.time() - start:.0f}s", flush=True)

    _, policy_params, _ = ppo.train(
        environment=env,
        progress_fn=progress,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=functools.partial(ppo_networks.make_ppo_networks, **network_settings),
        **params,
    )

    # the learned network weights, which the real attempter will load later
    CHECKPOINTS.mkdir(exist_ok=True)
    policy_path = CHECKPOINTS / f"pick_policy_seed{seed}"
    model.save_params(str(policy_path), policy_params)
    print(f"saved policy to {policy_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    main(parser.parse_args().seed)
