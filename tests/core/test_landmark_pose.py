import numpy as np
import pytest
from posecap_core import LandmarkPoseConverter, PoseCapError
from posecap_core.landmark_pose import NEUTRAL_TOE_DROP_DEGREES

_TOE_DROP_Y = 0.3 * float(np.sin(np.radians(NEUTRAL_TOE_DROP_DEGREES)))
_TOE_FORWARD_Z = 0.3 * float(np.cos(np.radians(NEUTRAL_TOE_DROP_DEGREES)))


def test_neutral_landmarks_convert_to_neutral_body_pose() -> None:
    converter = LandmarkPoseConverter()

    pose = converter.convert(_neutral_landmarks())

    assert np.allclose(pose.global_orient, [0.0, 0.0, 0.0], atol=1e-7)
    assert np.allclose(pose.body_pose, np.zeros((21, 3)), atol=1e-7)


def test_forward_left_arm_becomes_shoulder_rotation() -> None:
    landmarks = _neutral_landmarks()
    landmarks["left_elbow"] = (0.4, 1.0, 0.3)
    landmarks["left_wrist"] = (0.4, 1.0, 0.6)

    pose = LandmarkPoseConverter().convert(landmarks)

    left_shoulder = pose.body_pose[15]
    assert left_shoulder == pytest.approx([0.0, -np.pi / 2.0, 0.0])


def test_a_level_toe_reads_as_a_raised_foot_not_neutral() -> None:
    landmarks = _neutral_landmarks()
    landmarks["left_foot_index"] = (0.2, -2.0, 0.3)

    pose = LandmarkPoseConverter().convert(landmarks)

    left_ankle = pose.body_pose[6]
    assert float(np.linalg.norm(left_ankle)) == pytest.approx(
        np.radians(NEUTRAL_TOE_DROP_DEGREES), abs=1e-6
    )


def _thigh_direction(converter: LandmarkPoseConverter, landmarks) -> tuple[np.ndarray, float]:
    pose = converter.convert(landmarks)
    hip = pose.body_pose[0]
    angle = float(np.linalg.norm(hip))
    return np.asarray(hip) / angle if angle else np.zeros(3), angle


def test_a_foreshortened_thigh_aimed_at_the_camera_gets_its_depth_back() -> None:
    converter = LandmarkPoseConverter()
    converter.convert(_neutral_landmarks())  # standing frame: the thigh is 1.0 long
    squat = _neutral_landmarks()
    # True thigh: 1.0 long, 60 degrees forward. Estimator lost depth: z 0.5 -> 0.3.
    squat["left_knee"] = (0.2, -0.5, 0.3)
    squat["left_ankle"] = (0.2, -1.5, 0.3)
    squat["left_heel"] = (0.2, -1.5, 0.2)
    squat["left_foot_index"] = (0.2, -1.5 - _TOE_DROP_Y, 0.3 + _TOE_FORWARD_Z)

    _, angle = _thigh_direction(converter, squat)

    true_flexion = np.arctan2(np.sqrt(1.0 - 0.5**2), 0.5)
    assert angle == pytest.approx(true_flexion, abs=1e-6)


def test_a_full_length_thigh_is_left_as_measured() -> None:
    converter = LandmarkPoseConverter()
    stride = _neutral_landmarks()
    drop = float(np.cos(np.radians(30.0)))
    stride["left_knee"] = (0.2, -drop, 0.5)
    stride["left_ankle"] = (0.2, -drop - 1.0, 0.5)
    stride["left_heel"] = (0.2, -drop - 1.0, 0.4)
    stride["left_foot_index"] = (0.2, -drop - 1.0 - _TOE_DROP_Y, 0.5 + _TOE_FORWARD_Z)

    _, angle = _thigh_direction(converter, stride)

    assert angle == pytest.approx(np.radians(30.0), abs=1e-6)


def _neutral_landmarks() -> dict[str, tuple[float, float, float]]:
    return {
        "left_hip": (0.2, 0.0, 0.0),
        "right_hip": (-0.2, 0.0, 0.0),
        "left_shoulder": (0.4, 1.0, 0.0),
        "right_shoulder": (-0.4, 1.0, 0.0),
        "left_elbow": (0.7, 1.0, 0.0),
        "right_elbow": (-0.7, 1.0, 0.0),
        "left_wrist": (1.0, 1.0, 0.0),
        "right_wrist": (-1.0, 1.0, 0.0),
        "left_knee": (0.2, -1.0, 0.0),
        "right_knee": (-0.2, -1.0, 0.0),
        "left_ankle": (0.2, -2.0, 0.0),
        "right_ankle": (-0.2, -2.0, 0.0),
        "left_heel": (0.2, -2.0, -0.1),
        "right_heel": (-0.2, -2.0, -0.1),
        # A flat-footed stance: the toe sits NEUTRAL_TOE_DROP_DEGREES below the ankle.
        "left_foot_index": (0.2, -2.0 - _TOE_DROP_Y, _TOE_FORWARD_Z),
        "right_foot_index": (-0.2, -2.0 - _TOE_DROP_Y, _TOE_FORWARD_Z),
        "left_ear": (0.1, 1.55, 0.0),
        "right_ear": (-0.1, 1.55, 0.0),
        "mouth_left": (0.04, 1.4, 0.0),
        "mouth_right": (-0.04, 1.4, 0.0),
        "nose": (0.0, 1.5, 0.0),
    }


def test_missing_landmark_raises_domain_error() -> None:
    landmarks = _neutral_landmarks()
    del landmarks["left_wrist"]

    with pytest.raises(PoseCapError, match="missing landmark: left_wrist"):
        LandmarkPoseConverter().convert(landmarks)


def test_non_finite_landmark_raises_domain_error() -> None:
    landmarks = _neutral_landmarks()
    landmarks["nose"] = (float("nan"), 1.5, 0.0)

    with pytest.raises(PoseCapError, match="three finite numbers"):
        LandmarkPoseConverter().convert(landmarks)


def test_degenerate_landmarks_raise_domain_error() -> None:
    landmarks = {name: (0.0, 0.0, 0.0) for name in _neutral_landmarks()}

    with pytest.raises(PoseCapError, match="stable anatomical frame"):
        LandmarkPoseConverter().convert(landmarks)
