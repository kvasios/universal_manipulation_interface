"""
Extract a single video frame as PNG for designing a custom mask.json.

Usage:
  # From a session dir (defaults to gripper calibration video):
  python scripts/extract_mask_frame.py -s my_session

  # From a specific video:
  python scripts/extract_mask_frame.py -v path/to/raw_video.mp4

  # Also overlay the default mask.json for reference:
  python scripts/extract_mask_frame.py -s my_session --show_default_mask

  # Save standalone mask PNGs for quick inspection:
  python scripts/extract_mask_frame.py -s my_session --draw_masks
"""

import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import pathlib
import click
import av
import numpy as np
import cv2
from umi.common.cv_util import (
    draw_predefined_mask,
    get_default_mask_json_path,
    load_mask_json,
)


@click.command()
@click.option('-s', '--session_dir', default=None,
    help='Session directory. Defaults to the gripper calibration video inside it.')
@click.option('-v', '--video', default=None,
    help='Direct path to a raw_video.mp4. Overrides --session_dir.')
@click.option('-o', '--output', default=None,
    help='Output PNG path. Defaults to mask_reference_frame.png next to the video.')
@click.option('-f', '--frame_idx', type=int, default=None,
    help='Frame index to extract. Defaults to the middle frame.')
@click.option('-mj', '--mask_json', default=None,
    help='Mask JSON to preview. Defaults to umi/asset/mask.json.')
@click.option('--show_default_mask', is_flag=True, default=False,
    help='Also save mask overlays for the selected mask JSON.')
@click.option('--draw_masks', is_flag=True, default=False,
    help='Save standalone mask PNGs for quick evaluation.')
def main(session_dir, video, output, frame_idx, mask_json, show_default_mask, draw_masks):
    if video is not None:
        video_path = pathlib.Path(os.path.expanduser(video)).absolute()
    elif session_dir is not None:
        session = pathlib.Path(os.path.expanduser(session_dir)).absolute()
        demos_dir = session.joinpath('demos')
        gripper_cals = sorted(demos_dir.glob('gripper_calibration_*/raw_video.mp4'))
        if gripper_cals:
            video_path = gripper_cals[0]
        else:
            all_vids = sorted(demos_dir.glob('*/raw_video.mp4'))
            assert len(all_vids) > 0, f"No raw_video.mp4 found in {demos_dir}"
            video_path = all_vids[0]
    else:
        raise click.UsageError("Provide either --session_dir or --video.")

    assert video_path.is_file(), f"Video not found: {video_path}"
    if mask_json is None:
        mask_json = get_default_mask_json_path()
    else:
        mask_json = pathlib.Path(os.path.expanduser(mask_json)).absolute()
    assert mask_json.is_file(), f"Mask JSON not found: {mask_json}"
    mask_config = load_mask_json(mask_json)

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        total_frames = stream.frames
        if frame_idx is None:
            frame_idx = total_frames // 2

        for i, frame in enumerate(container.decode(stream)):
            if i == frame_idx:
                img = frame.to_ndarray(format='rgb24')
                break

    h, w = img.shape[:2]

    if output is None:
        output = video_path.parent.joinpath('mask_reference_frame.png')
    else:
        output = pathlib.Path(os.path.expanduser(output))

    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output), img_bgr)
    print(f"Saved reference frame to: {output}")
    print(f"  Video:      {video_path}")
    print(f"  Frame:      {frame_idx} / {total_frames}")
    print(f"  Resolution: {w} x {h}  (width x height)")
    print(f"  Mask JSON:  {mask_json}")
    print()
    print("Use these pixel coordinates when editing your mask.json.")
    print(f'  "resolution": [{h}, {w}]')

    if draw_masks:
        slam_mask = np.zeros((h, w), dtype=np.uint8)
        slam_mask = draw_predefined_mask(
            slam_mask, color=255, mirror=True, gripper=False, finger=True,
            mask_config=mask_config
        )
        slam_mask_path = output.parent.joinpath(output.stem + '_slam_mask.png')
        cv2.imwrite(str(slam_mask_path), slam_mask)

        training_mask = np.zeros((h, w), dtype=np.uint8)
        training_mask = draw_predefined_mask(
            training_mask, color=255, mirror=True, gripper=True, finger=False,
            mask_config=mask_config
        )
        training_mask_path = output.parent.joinpath(output.stem + '_training_mask.png')
        cv2.imwrite(str(training_mask_path), training_mask)

        aruco_mask = np.zeros((h, w), dtype=np.uint8)
        aruco_mask = draw_predefined_mask(
            aruco_mask, color=255, mirror=True, gripper=False, finger=False,
            mask_config=mask_config
        )
        aruco_mask_path = output.parent.joinpath(output.stem + '_aruco_mask.png')
        cv2.imwrite(str(aruco_mask_path), aruco_mask)

        print()
        print(f"Saved SLAM mask PNG to:     {slam_mask_path}")
        print(f"Saved training mask PNG to: {training_mask_path}")
        print(f"Saved ArUco mask PNG to:    {aruco_mask_path}")

    if show_default_mask:
        # SLAM mask (steps 02, 03): mirror + finger
        slam_overlay = img_bgr.copy()
        slam_overlay = draw_predefined_mask(slam_overlay, color=(0, 0, 255),
            mirror=True, gripper=False, finger=True, use_aa=True,
            mask_config=mask_config)
        blended = cv2.addWeighted(slam_overlay, 0.5, img_bgr, 0.5, 0)
        slam_path = output.parent.joinpath(
            output.stem + '_slam_mask_overlay' + output.suffix)
        cv2.imwrite(str(slam_path), blended)

        # Training mask (step 07, inference): mirror + gripper
        train_overlay = img_bgr.copy()
        train_overlay = draw_predefined_mask(train_overlay, color=(255, 0, 0),
            mirror=True, gripper=True, finger=False, use_aa=True,
            mask_config=mask_config)
        blended = cv2.addWeighted(train_overlay, 0.5, img_bgr, 0.5, 0)
        train_path = output.parent.joinpath(
            output.stem + '_training_mask_overlay' + output.suffix)
        cv2.imwrite(str(train_path), blended)

        print(f"\nSaved SLAM mask overlay to:     {slam_path}")
        print("  Red = mirror + finger regions (masked during SLAM, steps 02/03)")
        print(f"Saved training mask overlay to: {train_path}")
        print("  Blue = mirror + gripper regions (masked during training/inference, step 07)")


if __name__ == "__main__":
    main()
