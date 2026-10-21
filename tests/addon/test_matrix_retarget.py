"""Behavior tests for choosing the non-destructive live binding path."""

from types import SimpleNamespace

import numpy as np
from posecap_addon.matrix_retarget import requires_matrix_retarget
from posecap_core import BoundBone, PoseBinding

_IDENTITY = np.asarray([1.0, 0.0, 0.0, 0.0])
_JOINTS = ("left_shoulder", "left_elbow", "right_shoulder", "right_elbow")


class _IdentityMatrix:
    def __matmul__(self, position):
        return position


def _binding() -> PoseBinding:
    return PoseBinding({joint: BoundBone(joint, _IDENTITY, _IDENTITY) for joint in _JOINTS})


def _armature(*, rest_left_elbow, posed_left_elbow) -> SimpleNamespace:
    rest = {
        "left_shoulder": (0.15, 0.0, 1.5),
        "left_elbow": rest_left_elbow,
        "right_shoulder": (-0.15, 0.0, 1.5),
        "right_elbow": (-0.41, 0.0, 1.5),
    }
    posed = {**rest, "left_elbow": posed_left_elbow}
    return SimpleNamespace(
        matrix_world=_IdentityMatrix(),
        data=SimpleNamespace(
            bones={name: SimpleNamespace(head_local=head) for name, head in rest.items()}
        ),
        pose=SimpleNamespace(
            bones={name: SimpleNamespace(head=head) for name, head in posed.items()}
        ),
    )


def test_a_t_pose_rig_left_in_a_captured_pose_still_binds_directly() -> None:
    # A stopped stream leaves the last captured pose on the bones; restarting
    # must judge the rest pose, not the raised arm.
    armature = _armature(rest_left_elbow=(0.41, 0.0, 1.5), posed_left_elbow=(0.15, 0.0, 1.76))

    assert not requires_matrix_retarget(armature, _binding())


def test_a_steeply_drooped_bind_pose_uses_the_chain_aware_intermediary() -> None:
    armature = _armature(rest_left_elbow=(0.24, 0.0, 1.26), posed_left_elbow=(0.41, 0.0, 1.5))

    assert requires_matrix_retarget(armature, _binding())
