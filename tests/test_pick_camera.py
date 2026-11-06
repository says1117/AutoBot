import numpy as np
import pytest

# these parts run on JAX, which only the training venv has
jp = pytest.importorskip("jax.numpy")

from perception.pick_camera import GRASP_DISTANCE, CubeTracker, hide_cube_state

CUBE = np.array([[0.6, 0.1, 0.03]])


def test_cube_stays_put_until_the_gripper_arrives():
    tracker = CubeTracker(jp.array(CUBE))
    estimate = tracker.update(jp.array([[0.3, 0.0, 0.4]]))
    assert np.array(estimate) == pytest.approx(CUBE)
    assert not bool(tracker.grasped[0])


def test_cube_follows_the_gripper_after_the_grasp():
    tracker = CubeTracker(jp.array(CUBE))
    offset = np.array([[0.0, 0.0, GRASP_DISTANCE / 2]])
    tracker.update(jp.array(CUBE + offset))
    assert bool(tracker.grasped[0])

    # lift 20 cm: the cube keeps the same offset from the gripper as at the grasp
    lifted = CUBE + offset + np.array([[0.0, 0.0, 0.2]])
    estimate = tracker.update(jp.array(lifted))
    assert np.array(estimate) == pytest.approx(lifted - offset, abs=1e-6)


def test_tracker_handles_each_episode_on_its_own():
    starts = np.array([[0.6, 0.1, 0.03], [0.7, -0.1, 0.03]])
    tracker = CubeTracker(jp.array(starts))
    # the gripper reaches only the first cube
    tracker.update(jp.array([[0.6, 0.1, 0.03], [0.2, 0.0, 0.4]]))
    assert [bool(grasped) for grasped in tracker.grasped] == [True, False]


def test_no_exact_cube_value_reaches_the_policy():
    # 1000 + index makes every slot recognizable, so a missed value shows up by number
    obs = (1000.0 + jp.arange(66.0))[None]
    estimate = jp.array([[0.6, 0.1, 0.03]])
    target = jp.array([[0.5, 0.0, 0.3]])
    hidden = np.array(hide_cube_state(obs, estimate, target))[0]
    gripper = np.array(obs)[0, 31:34]

    assert hidden[9:12] == pytest.approx(estimate[0])
    assert hidden[12:16] == pytest.approx([1, 0, 0, 0])
    assert hidden[25:31] == pytest.approx(np.zeros(6))
    assert hidden[40:46] == pytest.approx([0, 1, 0, 0, 0, 1])
    assert hidden[46:49] == pytest.approx(np.array(estimate)[0] - gripper)
    assert hidden[49:52] == pytest.approx(np.array(target - estimate)[0])
    assert hidden[52:58] == pytest.approx(np.zeros(6))


def test_values_that_are_not_about_the_cube_are_left_alone():
    obs = (1000.0 + jp.arange(66.0))[None]
    hidden = np.array(hide_cube_state(obs, jp.zeros((1, 3)), jp.zeros((1, 3))))[0]
    untouched = [*range(9), *range(16, 25), *range(31, 40), *range(58, 66)]
    assert hidden[untouched] == pytest.approx(np.array(obs)[0, untouched])
