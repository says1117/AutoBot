import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from skills.registry import load_registry
from skills.runner import StubAttempter, run_skill

RESULTS_DIR = Path(__file__).parent / "results"


def evaluate(skill_name: str, args: dict, n_trials: int = 100) -> dict:
    # looks up skill and starts 2 tallies
    spec = load_registry()[skill_name]
    outcomes = Counter()
    total_attempts = 0

    for seed in range(n_trials):
        # runs skill using fake robot attempter that fixes the sequence of random rolls, making runs repeatable
        result = run_skill(spec, args, StubAttempter(seed=seed))
        # for each type of outcome, adds
        outcomes[result.outcome] += 1
        total_attempts += result.attempts

    return {
        "skill": skill_name,
        "trials": n_trials,
        # divides by # trials run, 3 decimal
        "success_rate": round(outcomes["success"] / n_trials, 3),
        "outcomes": dict(outcomes),
        "avg_attempts": round(total_attempts / n_trials, 3),
        "attempter": "stub",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def save(scorecard: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{scorecard['skill']}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(scorecard, indent=2))
    return path


if __name__ == "__main__":
    card = evaluate("pick", {"object_id": "red_cube"})
    print(json.dumps(card, indent=2))
    print("saved to", save(card))