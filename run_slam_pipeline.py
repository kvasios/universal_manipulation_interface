"""
Main script for UMI SLAM pipeline.

Usage:
  # Run with upstream example data (GoPro 10, default calibration):
  python run_slam_pipeline.py example_demo_session

  # Run with custom camera calibration and SLAM settings:
  python run_slam_pipeline.py my_session \\
      -c my_calibration/pipeline_calib \\
      -s my_calibration/my_camera_settings.yaml

Calibration directory must contain:
  - gopro_intrinsics_2_7k.json  (camera intrinsics for ArUco detection)
  - aruco_config.yaml           (ArUco marker dictionary and size map)
"""

import sys
import os

ROOT_DIR = os.path.dirname(__file__)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import pathlib
import click
import subprocess

DEFAULT_CALIBRATION_DIR = pathlib.Path(__file__).parent.joinpath('example', 'calibration')

# %%
@click.command()
@click.argument('session_dir', nargs=-1)
@click.option('-c', '--calibration_dir', type=str, default=None,
    help='Directory with gopro_intrinsics_2_7k.json and aruco_config.yaml. '
         f'Defaults to example/calibration (GoPro 10, 2704x2028).')
@click.option('-s', '--slam_settings', type=str, default=None,
    help='Host path to ORB_SLAM3 settings YAML for map creation and batch SLAM. '
         'If omitted, uses the GoPro 10 defaults baked into the docker image.')
def main(session_dir, calibration_dir, slam_settings):
    script_dir = pathlib.Path(__file__).parent.joinpath('scripts_slam_pipeline')
    if calibration_dir is None:
        calibration_dir = DEFAULT_CALIBRATION_DIR
    else:
        calibration_dir = pathlib.Path(os.path.expanduser(calibration_dir)).absolute()
    assert calibration_dir.is_dir(), f"Calibration dir not found: {calibration_dir}"

    camera_intrinsics = calibration_dir.joinpath('gopro_intrinsics_2_7k.json')
    aruco_config = calibration_dir.joinpath('aruco_config.yaml')
    assert camera_intrinsics.is_file(), \
        f"Missing camera intrinsics: {camera_intrinsics}"
    assert aruco_config.is_file(), \
        f"Missing ArUco config: {aruco_config}"

    if slam_settings is not None:
        slam_settings = pathlib.Path(os.path.expanduser(slam_settings)).absolute()
        assert slam_settings.is_file(), f"SLAM settings not found: {slam_settings}"

    print("=" * 60)
    print("Pipeline configuration")
    print(f"  Calibration dir : {calibration_dir}")
    print(f"  Camera intrinsics: {camera_intrinsics}")
    print(f"  ArUco config     : {aruco_config}")
    print(f"  SLAM settings    : {slam_settings or '(docker default: GoPro 10)'}")
    print("=" * 60)

    for session in session_dir:
        session = pathlib.Path(os.path.expanduser(session)).absolute()

        print("############## 00_process_videos #############")
        script_path = script_dir.joinpath("00_process_videos.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 01_extract_gopro_imu ###########")
        script_path = script_dir.joinpath("01_extract_gopro_imu.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 02_create_map ###########")
        script_path = script_dir.joinpath("02_create_map.py")
        assert script_path.is_file()
        demo_dir = session.joinpath('demos')
        mapping_dir = demo_dir.joinpath('mapping')
        assert mapping_dir.is_dir()
        map_path = mapping_dir.joinpath('map_atlas.osa')
        if not map_path.is_file():
            cmd = [
                'python', str(script_path),
                '--input_dir', str(mapping_dir),
                '--map_path', str(map_path)
            ]
            if slam_settings is not None:
                cmd.extend(['--slam_settings', str(slam_settings)])
            result = subprocess.run(cmd)
            assert result.returncode == 0
            assert map_path.is_file()

        print("############# 03_batch_slam ###########")
        script_path = script_dir.joinpath("03_batch_slam.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input_dir', str(demo_dir),
            '--map_path', str(map_path)
        ]
        if slam_settings is not None:
            cmd.extend(['--slam_settings', str(slam_settings)])
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 04_detect_aruco ###########")
        script_path = script_dir.joinpath("04_detect_aruco.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input_dir', str(demo_dir),
            '--camera_intrinsics', str(camera_intrinsics),
            '--aruco_yaml', str(aruco_config)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 05_run_calibrations ###########")
        script_path = script_dir.joinpath("05_run_calibrations.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

        print("############# 06_generate_dataset_plan ###########")
        script_path = script_dir.joinpath("06_generate_dataset_plan.py")
        assert script_path.is_file()
        cmd = [
            'python', str(script_path),
            '--input', str(session)
        ]
        result = subprocess.run(cmd)
        assert result.returncode == 0

## %%
if __name__ == "__main__":
    main()
