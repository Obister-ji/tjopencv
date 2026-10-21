"""Keep a pelvis-locked character's feet on the floor it stands on at rest.

Poses apply pelvis-locked, so a bent leg lifts the feet off the floor and a
foot pitched toes-up pushes the heel through it. Given the leg chain's rest
matrices and a few sole contact points, this computes how far the pelvis must
move along the up axis so the lowest contact point returns to its rest height.
Pure forward kinematics: the Blender adapter supplies the rest data and the
bone rotations it just wrote.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .rotation import FloatArray

FOOT_CHAIN_JOINTS = {
    side: (f"{side}_hip", f"{side}_knee", f"{side}_ankle", f"{side}_foot")
    for side in ("left", "right")
}


@dataclass(frozen=True)
class GroundBone:
    """One bone of the leg chains, in world space at rest.

    ``parent`` names the nearest ancestor that is also in the chain, or
    ``None`` for the pelvis. Bones between them are assumed unposed.
    """

    name: str
    parent: str | None
    rest_matrix: FloatArray


@dataclass(frozen=True)
class GroundRig:
    """The leg chains plus the sole points that touch the floor at rest."""

    bones: tuple[GroundBone, ...]
    contacts: tuple[tuple[str, FloatArray], ...]
    up: FloatArray


def ground_lift(rig: GroundRig, rotations: Mapping[str, Sequence[float] | FloatArray]) -> float:
    """Distance along ``rig.up`` that puts the lowest contact back on its rest floor.

    ``rotations`` maps bone names to the local quaternions (w, x, y, z) now on
    the bones; a bone without one is treated as unposed. Positive lifts the
    character, negative lowers it (a squat, a raised knee).
    """
    rest = {bone.name: np.asarray(bone.rest_matrix, dtype=np.float64) for bone in rig.bones}
    posed: dict[str, np.ndarray] = {}
    for bone in rig.bones:
        local = _rotation_matrix(rotations.get(bone.name))
        if bone.parent is None:
            posed[bone.name] = rest[bone.name] @ local
        else:
            posed[bone.name] = (
                posed[bone.parent] @ np.linalg.inv(rest[bone.parent]) @ rest[bone.name] @ local
            )
    up = np.asarray(rig.up, dtype=np.float64)
    rest_floor = min(float(up @ np.asarray(point)) for _, point in rig.contacts)
    lowest = min(
        float(up @ _transform(posed[name] @ np.linalg.inv(rest[name]), point))
        for name, point in rig.contacts
    )
    return rest_floor - lowest


def _rotation_matrix(quaternion: Sequence[float] | FloatArray | None) -> np.ndarray:
    matrix = np.eye(4)
    if quaternion is None:
        return matrix
    w, x, y, z = (float(value) for value in quaternion)
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm == 0.0:
        return matrix
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    matrix[:3, :3] = (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )
    return matrix


def _transform(matrix: np.ndarray, point: FloatArray) -> np.ndarray:
    return (matrix @ np.append(np.asarray(point, dtype=np.float64), 1.0))[:3]
