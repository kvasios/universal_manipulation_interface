#!/usr/bin/env python3
"""
Generate an ORB-SLAM3 monocular-inertial settings YAML from OpenICC outputs.

This script is intentionally standalone so calibration conversion stays explicit
and reviewable rather than being hidden inside the UMI pipeline.

Typical usage:

python scripts/gen_orbslam3_yaml_from_openicc.py \
    --camera-calib-json gopro_cal_data/cam/cam_calib_GX011912_fi_2.0.json \
    --cam-imu-calib-json gopro_cal_data/cam_imu/cam_imu_calib_result_GX011911.json \
    --template ORB_SLAM3/Examples/Monocular-Inertial/gopro10_maxlens_fisheye_setting_v1.yaml \
    --output custom_gopro9_fisheye_1352x1014.yaml

Notes:
- OpenICC camera calibration JSON is expected to be the `FISHEYE` model.
- OpenICC `cam_imu_calib_result_*.json` stores `T_i_c` (IMU to camera). ORB-SLAM3
  expects `Tbc` / `IMU.T_b_c1` (camera to IMU/body), so this script inverts the
  transform by default.
- Some GoPro example YAMLs in the ORB-SLAM3 fork contain an extra axis permutation
  comment. That is left as an explicit option here instead of being silently applied.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


KEEP_KEYS = [
    "File.version",
    "Camera.RGB",
    "IMU.NoiseGyro",
    "IMU.NoiseAcc",
    "IMU.GyroWalk",
    "IMU.AccWalk",
    "IMU.Frequency",
    "IMU.InsertKFsWhenLost",
    "ORBextractor.nFeatures",
    "ORBextractor.scaleFactor",
    "ORBextractor.nLevels",
    "ORBextractor.iniThFAST",
    "ORBextractor.minThFAST",
    "System.thFarPoints",
    "System.LoadAtlasFromFile",
    "System.SaveAtlasToFile",
    "Viewer.KeyFrameSize",
    "Viewer.KeyFrameLineWidth",
    "Viewer.GraphLineWidth",
    "Viewer.PointSize",
    "Viewer.CameraSize",
    "Viewer.CameraLineWidth",
    "Viewer.ViewpointX",
    "Viewer.ViewpointY",
    "Viewer.ViewpointZ",
    "Viewer.ViewpointF",
    "Viewer.imageViewScale",
]


PERMUTATIONS = {
    "none": np.eye(4, dtype=np.float64),
    # Matches the comment included in the GoPro v1 YAML examples.
    "gopro_v1": np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate ORB-SLAM3 YAML from OpenICC calibration outputs."
    )
    parser.add_argument("--camera-calib-json", required=True, help="OpenICC camera calibration JSON.")
    parser.add_argument(
        "--cam-imu-calib-json",
        required=True,
        help="OpenICC camera-IMU calibration JSON, typically cam_imu_calib_result_*.json.",
    )
    parser.add_argument(
        "--template",
        required=True,
        help="Template ORB-SLAM3 YAML to borrow non-calibration settings from.",
    )
    parser.add_argument("--output", required=True, help="Output YAML path.")
    parser.add_argument(
        "--output-width",
        type=int,
        default=None,
        help="Optional target width. Defaults to the camera calibration width.",
    )
    parser.add_argument(
        "--output-height",
        type=int,
        default=None,
        help="Optional target height. Defaults to the camera calibration height.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional output fps. Defaults to the camera calibration fps.",
    )
    parser.add_argument(
        "--invert-extrinsics",
        action="store_true",
        default=True,
        help="Invert OpenICC T_i_c to camera-to-IMU. Enabled by default.",
    )
    parser.add_argument(
        "--no-invert-extrinsics",
        dest="invert_extrinsics",
        action="store_false",
        help="Use the OpenICC transform as-is.",
    )
    parser.add_argument(
        "--axis-permutation",
        choices=sorted(PERMUTATIONS),
        default="none",
        help="Optional axis permutation to apply after the camera-to-IMU transform.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_template_scalar_lines(lines: Iterable[str]) -> Dict[str, str]:
    values: Dict[str, str] = {}
    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ": " not in stripped:
            continue
        key, value = stripped.split(": ", 1)
        if key in KEEP_KEYS:
            values[key] = value
    return values


def detect_template_style(template_text: str) -> Tuple[str, str]:
    if "Camera1.fx:" in template_text:
        camera_prefix = "Camera1"
    elif "Camera.fx:" in template_text:
        camera_prefix = "Camera"
    else:
        raise ValueError("Could not detect camera key style from template YAML.")

    if "IMU.T_b_c1:" in template_text:
        extrinsic_key = "IMU.T_b_c1"
    elif "\nTbc:" in template_text or template_text.startswith("Tbc:"):
        extrinsic_key = "Tbc"
    else:
        raise ValueError("Could not detect extrinsic key style from template YAML.")

    return camera_prefix, extrinsic_key


def quaternion_to_rotation_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_from_openicc(cam_imu_data: dict, invert_extrinsics: bool, axis_permutation: str) -> np.ndarray:
    quat = cam_imu_data["q_i_c"]
    trans = cam_imu_data["t_i_c"]

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_to_rotation_matrix(
        quat["w"], quat["x"], quat["y"], quat["z"]
    )
    transform[:3, 3] = np.array([trans["x"], trans["y"], trans["z"]], dtype=np.float64)

    if invert_extrinsics:
        transform = np.linalg.inv(transform)

    transform = PERMUTATIONS[axis_permutation] @ transform
    return transform


def camera_params_from_openicc(camera_data: dict) -> Dict[str, float]:
    if camera_data["intrinsic_type"] != "FISHEYE":
        raise ValueError(
            f"Only OpenICC FISHEYE camera calibrations are supported right now, "
            f"got {camera_data['intrinsic_type']!r}."
        )

    intr = camera_data["intrinsics"]
    focal = float(intr["focal_length"])
    aspect_ratio = float(intr["aspect_ratio"])

    return {
        "width": int(camera_data["image_width"]),
        "height": int(camera_data["image_height"]),
        "fps": float(camera_data["fps"]),
        "fx": focal,
        "fy": focal * aspect_ratio,
        "cx": float(intr["principal_pt_x"]),
        "cy": float(intr["principal_pt_y"]),
        "k1": float(intr["radial_distortion_1"]),
        "k2": float(intr["radial_distortion_2"]),
        "k3": float(intr["radial_distortion_3"]),
        "k4": float(intr["radial_distortion_4"]),
    }


def resize_intrinsics(
    params: Dict[str, float], target_width: int, target_height: int
) -> Dict[str, float]:
    in_width = params["width"]
    in_height = params["height"]
    out_width = target_width
    out_height = target_height

    if in_width == out_width and in_height == out_height:
        return dict(params)

    # Match the UMI repo's fisheye intrinsics conversion assumption:
    # preserve the vertical field of view and assume only symmetric horizontal
    # crop/pad differences.
    scale = out_height / in_height
    out = dict(params)
    out["width"] = out_width
    out["height"] = out_height
    out["fx"] = params["fx"] * scale
    out["fy"] = params["fy"] * scale
    out["cx"] = (params["cx"] - (in_width / 2.0)) * scale + (out_width / 2.0)
    out["cy"] = params["cy"] * scale
    return out


def format_float(value: float) -> str:
    if math.isfinite(value) and abs(value - round(value)) < 1e-9:
        return f"{value:.1f}"
    return f"{value:.12g}"


def format_int_like(value: float) -> str:
    return str(int(round(value)))


def format_matrix_row_major(transform: np.ndarray) -> List[str]:
    flat = transform.reshape(-1)
    return [format_float(float(x)) for x in flat]


def emit_yaml(
    output_path: Path,
    camera_prefix: str,
    extrinsic_key: str,
    template_values: Dict[str, str],
    camera_params: Dict[str, float],
    transform: np.ndarray,
    meta: Dict[str, str],
) -> None:
    cam_key = lambda suffix: f"{camera_prefix}.{suffix}"
    matrix_indent = "    " if extrinsic_key.startswith("IMU.") else "   "

    def inherited(key: str, default: str) -> str:
        return template_values.get(key, default)

    data_values = format_matrix_row_major(transform)

    lines = [
        "%YAML:1.0",
        "",
        "#--------------------------------------------------------------------------------------------",
        "# Auto-generated from OpenICC calibration outputs.",
        "#--------------------------------------------------------------------------------------------",
    ]

    file_version = template_values.get("File.version")
    if file_version is not None:
        lines.append(f'File.version: {file_version}')

    lines.extend(
        [
            'Camera.type: "KannalaBrandt8"',
            f"# Source camera calib: {meta['camera_calib_json']}",
            f"# Source cam-imu calib: {meta['cam_imu_calib_json']}",
            f"# Extrinsics: {'inverse(T_i_c)' if meta['invert_extrinsics'] == 'true' else 'T_i_c as-is'}",
            f"# Axis permutation: {meta['axis_permutation']}",
            f"{cam_key('fx')}: {format_float(camera_params['fx'])}",
            f"{cam_key('fy')}: {format_float(camera_params['fy'])}",
            f"{cam_key('cx')}: {format_float(camera_params['cx'])}",
            f"{cam_key('cy')}: {format_float(camera_params['cy'])}",
            "",
            f"{cam_key('k1')}: {format_float(camera_params['k1'])}",
            f"{cam_key('k2')}: {format_float(camera_params['k2'])}",
            f"{cam_key('k3')}: {format_float(camera_params['k3'])}",
            f"{cam_key('k4')}: {format_float(camera_params['k4'])}",
            "",
            "# Camera resolution",
            f"Camera.width: {camera_params['width']}",
            f"Camera.height: {camera_params['height']}",
            "",
            "# Camera frames per second",
            f"Camera.fps: {format_int_like(camera_params['fps'])}",
            "",
            "# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)",
            f"Camera.RGB: {inherited('Camera.RGB', '1')}",
            "",
            "# Transformation from camera to imu (body frame)",
            f"{extrinsic_key}: !!opencv-matrix",
            f"{matrix_indent}rows: 4",
            f"{matrix_indent}cols: 4",
            f"{matrix_indent}dt: f",
            f"{matrix_indent}data: [{', '.join(data_values)}]",
            "",
            "# IMU noise -> copied from template unless changed manually later",
            f"IMU.NoiseGyro: {inherited('IMU.NoiseGyro', '0.0015')}",
            f"IMU.NoiseAcc: {inherited('IMU.NoiseAcc', '0.017')}",
            f"IMU.GyroWalk: {inherited('IMU.GyroWalk', '5.0e-5')}",
            f"IMU.AccWalk: {inherited('IMU.AccWalk', '0.0055')}",
            f"IMU.Frequency: {inherited('IMU.Frequency', '200.0')}",
        ]
    )

    if "IMU.InsertKFsWhenLost" in template_values:
        lines.append(f"IMU.InsertKFsWhenLost: {template_values['IMU.InsertKFsWhenLost']}")

    lines.extend(
        [
            "",
            "#--------------------------------------------------------------------------------------------",
            "# ORB Parameters",
            "#--------------------------------------------------------------------------------------------",
            f"ORBextractor.nFeatures: {inherited('ORBextractor.nFeatures', '2500')}",
            f"ORBextractor.scaleFactor: {inherited('ORBextractor.scaleFactor', '1.2')}",
            f"ORBextractor.nLevels: {inherited('ORBextractor.nLevels', '10')}",
            f"ORBextractor.iniThFAST: {inherited('ORBextractor.iniThFAST', '20')}",
            f"ORBextractor.minThFAST: {inherited('ORBextractor.minThFAST', '7')}",
        ]
    )

    if "System.thFarPoints" in template_values:
        lines.append(f"System.thFarPoints: {template_values['System.thFarPoints']}")
    if "System.LoadAtlasFromFile" in template_values:
        lines.append(f"System.LoadAtlasFromFile: {template_values['System.LoadAtlasFromFile']}")
    if "System.SaveAtlasToFile" in template_values:
        lines.append(f"System.SaveAtlasToFile: {template_values['System.SaveAtlasToFile']}")

    lines.extend(
        [
            "",
            "#--------------------------------------------------------------------------------------------",
            "# Viewer Parameters",
            "#--------------------------------------------------------------------------------------------",
            f"Viewer.KeyFrameSize: {inherited('Viewer.KeyFrameSize', '0.05')}",
            f"Viewer.KeyFrameLineWidth: {inherited('Viewer.KeyFrameLineWidth', '1.0')}",
            f"Viewer.GraphLineWidth: {inherited('Viewer.GraphLineWidth', '0.9')}",
            f"Viewer.PointSize: {inherited('Viewer.PointSize', '2.0')}",
            f"Viewer.CameraSize: {inherited('Viewer.CameraSize', '0.08')}",
            f"Viewer.CameraLineWidth: {inherited('Viewer.CameraLineWidth', '3.0')}",
            f"Viewer.ViewpointX: {inherited('Viewer.ViewpointX', '0.0')}",
            f"Viewer.ViewpointY: {inherited('Viewer.ViewpointY', '-0.7')}",
            f"Viewer.ViewpointZ: {inherited('Viewer.ViewpointZ', '-3.5')}",
            f"Viewer.ViewpointF: {inherited('Viewer.ViewpointF', '500.0')}",
        ]
    )

    if "Viewer.imageViewScale" in template_values:
        lines.append(f"Viewer.imageViewScale: {template_values['Viewer.imageViewScale']}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()

    camera_calib_path = Path(args.camera_calib_json).expanduser().resolve()
    cam_imu_calib_path = Path(args.cam_imu_calib_json).expanduser().resolve()
    template_path = Path(args.template).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    camera_data = load_json(camera_calib_path)
    cam_imu_data = load_json(cam_imu_calib_path)
    template_text = template_path.read_text(encoding="utf-8")
    template_values = parse_template_scalar_lines(template_text.splitlines())
    camera_prefix, extrinsic_key = detect_template_style(template_text)

    camera_params = camera_params_from_openicc(camera_data)
    target_width = args.output_width or camera_params["width"]
    target_height = args.output_height or camera_params["height"]
    camera_params = resize_intrinsics(camera_params, target_width, target_height)
    camera_params["fps"] = args.fps if args.fps is not None else camera_params["fps"]

    transform = transform_from_openicc(
        cam_imu_data,
        invert_extrinsics=args.invert_extrinsics,
        axis_permutation=args.axis_permutation,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    emit_yaml(
        output_path=output_path,
        camera_prefix=camera_prefix,
        extrinsic_key=extrinsic_key,
        template_values=template_values,
        camera_params=camera_params,
        transform=transform,
        meta={
            "camera_calib_json": str(camera_calib_path),
            "cam_imu_calib_json": str(cam_imu_calib_path),
            "invert_extrinsics": str(args.invert_extrinsics).lower(),
            "axis_permutation": args.axis_permutation,
        },
    )

    print(f"Wrote ORB-SLAM3 YAML to {output_path}")


if __name__ == "__main__":
    main()
