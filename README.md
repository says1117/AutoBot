Source code going to be pushed out soon! :)
## Install

Needs Linux (or WSL) with Python 3.12, plus an NVIDIA GPU to run the arm. The first run also downloads the robot model and compiles for a few minutes.

```bash
git clone git@github.com:says1117/AutoBot.git
cd AutoBot
python3 -m venv .venv
.venv/bin/pip install -c requirements-lock.txt -e ".[sim,dev]"
```

The planner and the photo check use NVIDIA NIM, so set your key first:

```bash
export LLM_API_KEY=your_key_here
```

## Run

```bash
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
.venv/bin/python -m planner.demo "Pick up the cube."
.venv/bin/python -m evals.watch_pick
.venv/bin/python -m evals.eval_pick_camera --object red_cube
```

## Test

The tests need no GPU and no API key:

```bash
.venv/bin/python -m pytest
```

Without a GPU, install with `pip install -e ".[dev]"` instead. The JAX tests are then skipped.
