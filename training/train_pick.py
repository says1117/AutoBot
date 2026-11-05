import functools
import time
from pathlib import Path

from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import manipulation_params

ENV_NAME = "PandaPickCube"
POLICY_PATH = Path(__file__).parent.parent / "checkpoints" / "pick_policy"


def main():
    # the simulated arm, cube, and reward
    config = registry.get_default_config(ENV_NAME)
    env = registry.load(ENV_NAME, config=config)

    # PPO settings; network sizes are passed separately from the rest
    params = manipulation_params.brax_ppo_config(ENV_NAME).to_dict()
    network_settings = params.pop("network_factory")
    params.update(num_evals=11, num_resets_per_eval=0)

    start = time.time()

    # brax calls this after each evaluation
    def progress(step, metrics):
        reward = float(metrics["eval/episode_reward"])
        print(f"step {step:,} | reward {reward:.1f} | {time.time() - start:.0f}s", flush=True)

    _, policy_params, _ = ppo.train(
        environment=env,
        progress_fn=progress,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=functools.partial(ppo_networks.make_ppo_networks, **network_settings),
        **params,
    )

    # the learned network weights, which the real attempter will load later
    POLICY_PATH.parent.mkdir(exist_ok=True)
    model.save_params(str(POLICY_PATH), policy_params)
    print(f"saved policy to {POLICY_PATH}", flush=True)


if __name__ == "__main__":
    main()
