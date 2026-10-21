"""Pure mapping from SMPL-X pose instructions to an untouched target rig.

The binding is calculated during setup by a Blender-facing adapter.  On the
live path this module only rewrites an immutable :class:`PoseApplication`, so
it neither knows about nor mutates Blender data.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .application import BoneRotation, PoseApplication
from .errors import PoseCapError
from .rotation import IDENTITY_QUATERNION as _IDENTITY_QUATERNION
from .rotation import ZERO_ANGLE, FloatArray, quaternion_multiply


@dataclass(frozen=True)
class BoundBone:
    """One SMPL-X joint mapped to an original bone on the target rig.

    ``compensation_quaternion`` changes from the SMPL-X local basis into the
    target bone's local basis.  ``neutral_quaternion`` is the target pose that
    represents a neutral capture, which lets reset honour an untouched rig's
    current rest pose rather than assuming a global identity.
    """

    target_bone_name: str
    compensation_quaternion: FloatArray
    neutral_quaternion: FloatArray

    def __post_init__(self) -> None:
        if not self.target_bone_name:
            raise PoseCapError("bound target bone name cannot be empty")
        object.__setattr__(
            self,
            "compensation_quaternion",
            _normalized_readonly(self.compensation_quaternion),
        )
        object.__setattr__(
            self,
            "neutral_quaternion",
            _normalized_readonly(self.neutral_quaternion),
        )


@dataclass(frozen=True)
class PoseBinding:
    """Immutable lookup from SMPL-X joint names to original target bones."""

    bones: Mapping[str, BoundBone]

    def __post_init__(self) -> None:
        copied = dict(self.bones)
        if any(not name for name in copied):
            raise PoseCapError("SMPL-X joint name cannot be empty")
        targets = [bone.target_bone_name for bone in copied.values()]
        if len(targets) != len(set(targets)):
            raise PoseCapError("each target bone can be bound only once")
        object.__setattr__(self, "bones", MappingProxyType(copied))

    def source_previous_quaternions(
        self,
        target_previous_quaternions: Mapping[str, FloatArray],
    ) -> dict[str, FloatArray]:
        """Express target-writer history back in SMPL-X joint coordinates."""
        previous: dict[str, FloatArray] = {}
        for source_name, bone in self.bones.items():
            target_quaternion = target_previous_quaternions.get(bone.target_bone_name)
            if target_quaternion is not None:
                previous[source_name] = _source_quaternion(target_quaternion, bone)
        return previous


def compensation_from_rest_orientations(
    source_rest_orientation: FloatArray,
    target_rest_orientation: FloatArray,
) -> FloatArray:
    """Build a target-basis compensation from accumulated rest orientations.

    Both inputs must describe the same global reference frame. The Blender
    adapter therefore supplies each bone's accumulated rest orientation and
    includes any armature-object rotation before calling this function. The
    resulting quaternion maps a streamed source rotation through
    ``compensation @ source @ compensation^-1``.
    """
    source = _normalized_readonly(source_rest_orientation)
    target = _normalized_readonly(target_rest_orientation)
    return _normalized_readonly(quaternion_multiply(_conjugate(target), source))


@dataclass(frozen=True)
class RestCorrection:
    """A bone's rest orientation after swinging a drooping arm into a T-pose.

    ``rest_orientation`` is the world orientation the bone has once corrected;
    ``neutral_quaternion`` is the target-local pose that realises it, so a
    neutral capture lands the arm on the T-pose instead of its bind droop.
    """

    rest_orientation: FloatArray
    neutral_quaternion: FloatArray


def t_pose_arm_corrections(
    arm_chains: Mapping[str, Sequence[tuple[str, str]]],
    targets: Mapping[str, Sequence[float] | FloatArray],
    heads: Mapping[str, Sequence[float] | FloatArray],
    rest_orientations: Mapping[str, FloatArray],
) -> dict[str, RestCorrection]:
    """Swing each arm chain onto its T-pose direction, carrying child bones along.

    ``arm_chains`` maps a side key to ordered (bone, child) pairs; ``targets``
    gives that side's world T-pose direction; ``heads`` and ``rest_orientations``
    are world-space rest data per bone. Bones in the chain, and the hand that
    ends it, get a correction; a chain that cannot be measured is left alone.
    """
    corrections: dict[str, RestCorrection] = {}
    for side, chain in arm_chains.items():
        target = np.asarray(targets[side], dtype=np.float64)
        parent = np.asarray(_IDENTITY_QUATERNION)
        pending: dict[str, RestCorrection] = {}
        for bone, child in chain:
            if bone not in rest_orientations or bone not in heads or child not in heads:
                pending = {}
                break
            direction = np.asarray(heads[child], dtype=np.float64) - np.asarray(
                heads[bone], dtype=np.float64
            )
            if float(np.linalg.norm(direction)) < ZERO_ANGLE:
                pending = {}
                break
            swing = _rotation_between(_rotate(parent, direction), target)
            accumulated = quaternion_multiply(swing, parent)
            rest = np.asarray(rest_orientations[bone], dtype=np.float64)
            neutral = quaternion_multiply(
                _conjugate(rest),
                quaternion_multiply(_conjugate(parent), quaternion_multiply(accumulated, rest)),
            )
            pending[bone] = RestCorrection(
                rest_orientation=_normalized_readonly(quaternion_multiply(accumulated, rest)),
                neutral_quaternion=_normalized_readonly(neutral),
            )
            parent = accumulated
        if not pending:
            continue
        hand = chain[-1][1]
        if hand in rest_orientations:
            hand_rest = quaternion_multiply(
                parent, np.asarray(rest_orientations[hand], dtype=np.float64)
            )
            pending[hand] = RestCorrection(
                rest_orientation=_normalized_readonly(hand_rest),
                neutral_quaternion=_IDENTITY_QUATERNION,
            )
        corrections.update(pending)
    return corrections


def apply_binding(plan: PoseApplication, binding: PoseBinding) -> PoseApplication:
    """Rewrite a SMPL-X pose plan to the target rig's original bone names.

    Unmapped joints are intentionally omitted.  This permits a body-only
    binding to stream while reporting missing optional fingers during setup.
    """
    clear_bones = _bound_clear_bones(plan.clear_bones, binding)
    neutral_rotations = tuple(
        BoneRotation(
            bone_name=bone.target_bone_name,
            quaternion=bone.neutral_quaternion,
        )
        for bone in _bound_bones_for_clear(plan.clear_bones, binding)
    )
    rotations = tuple(
        _bound_rotation(rotation, binding.bones[rotation.bone_name])
        for rotation in plan.rotations
        if rotation.bone_name in binding.bones
    )
    return PoseApplication(
        clear_bones=clear_bones,
        rotations=rotations,
        world_offset=plan.world_offset,
        neutral_rotations=neutral_rotations,
    )


def _bound_clear_bones(
    source_clear_bones: frozenset[str] | None,
    binding: PoseBinding,
) -> frozenset[str]:
    return frozenset(
        bone.target_bone_name for bone in _bound_bones_for_clear(source_clear_bones, binding)
    )


def _bound_bones_for_clear(
    source_clear_bones: frozenset[str] | None,
    binding: PoseBinding,
) -> tuple[BoundBone, ...]:
    source_names = binding.bones if source_clear_bones is None else source_clear_bones
    return tuple(
        binding.bones[source_name] for source_name in source_names if source_name in binding.bones
    )


def _bound_rotation(rotation: BoneRotation, bone: BoundBone) -> BoneRotation:
    compensated = quaternion_multiply(
        bone.compensation_quaternion,
        quaternion_multiply(rotation.quaternion, _conjugate(bone.compensation_quaternion)),
    )
    quaternion = quaternion_multiply(bone.neutral_quaternion, compensated)
    quaternion = _normalized_readonly(quaternion)
    return BoneRotation(bone_name=bone.target_bone_name, quaternion=quaternion)


def _source_quaternion(target_quaternion: FloatArray, bone: BoundBone) -> FloatArray:
    without_neutral = quaternion_multiply(
        _conjugate(bone.neutral_quaternion),
        target_quaternion,
    )
    return quaternion_multiply(
        _conjugate(bone.compensation_quaternion),
        quaternion_multiply(without_neutral, bone.compensation_quaternion),
    )


def _rotation_between(source: FloatArray, target: FloatArray) -> FloatArray:
    a = source / np.linalg.norm(source)
    b = target / np.linalg.norm(target)
    cosine = float(np.dot(a, b))
    if cosine < -1.0 + 1e-9:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return np.array([0.0, *axis])
    quaternion = np.array([1.0 + cosine, *np.cross(a, b)])
    return quaternion / np.linalg.norm(quaternion)


def _rotate(quaternion: FloatArray, vector: FloatArray) -> FloatArray:
    rotated = quaternion_multiply(
        quaternion_multiply(quaternion, np.array([0.0, *vector])), _conjugate(quaternion)
    )
    return rotated[1:]


def _conjugate(quaternion: FloatArray) -> FloatArray:
    result = np.asarray(quaternion, dtype=np.float64).copy()
    result[1:] *= -1.0
    return result


def _normalized_readonly(quaternion: FloatArray) -> FloatArray:
    result = np.asarray(quaternion, dtype=np.float64)
    if result.shape != (4,):
        raise PoseCapError("quaternion must have four components")
    norm = float(np.linalg.norm(result))
    if norm < ZERO_ANGLE:
        raise PoseCapError("quaternion cannot have zero norm")
    normalized = (result / norm).copy()
    normalized.setflags(write=False)
    return normalized
