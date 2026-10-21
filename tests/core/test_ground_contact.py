"""Behavior tests for keeping a pelvis-locked character's feet on the floor."""

import numpy as np
import pytest
from posecap_core import GroundBone, GroundRig, axis_angle_to_quaternion, ground_lift


def _at(z: float) -> np.ndarray:
    matrix = np.eye(4)
    matrix[2, 3] = z
    return matrix


def _leg_rig() -> GroundRig:
    """A straight leg: pelvis and hip at 1.0, knee at 0.5, ankle on the floor."""
    return GroundRig(
        bones=(
            GroundBone("pelvis", None, _at(1.0)),
            GroundBone("hip", "pelvis", _at(1.0)),
            GroundBone("knee", "hip", _at(0.5)),
            GroundBone("ankle", "knee", _at(0.0)),
        ),
        contacts=(
            ("ankle", np.array([0.0, 0.0, 0.0])),
            ("ankle", np.array([0.0, -0.2, 0.0])),
        ),
        up=np.array([0.0, 0.0, 1.0]),
    )


def test_a_rest_pose_needs_no_lift() -> None:
    assert ground_lift(_leg_rig(), {}) == pytest.approx(0.0)


def test_a_bent_knee_lowers_the_body_until_the_foot_touches_down() -> None:
    bent = axis_angle_to_quaternion(np.array([np.pi / 2.0, 0.0, 0.0]))

    # The shin swings level at knee height (0.5) and the foot hangs from it, toe
    # 0.2 below, so the body drops until that toe reaches the floor.
    assert ground_lift(_leg_rig(), {"knee": bent}) == pytest.approx(-0.3)


def test_a_toes_down_foot_lifts_the_body_off_the_floor() -> None:
    pointed = axis_angle_to_quaternion(np.array([np.pi / 6.0, 0.0, 0.0]))

    # Forward is -Y here: the toe point (0.2 forward) dips 0.2 * sin(30 deg).
    assert ground_lift(_leg_rig(), {"ankle": pointed}) == pytest.approx(0.1)
