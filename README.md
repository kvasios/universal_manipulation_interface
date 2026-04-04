# Universal Manipulation Interface

[[Project page]](https://umi-gripper.github.io/)
[[Paper]](https://umi-gripper.github.io/#paper)
[[Hardware Guide]](https://docs.google.com/document/d/1TPYwV9sNVPAi0ZlAupDMkXZ4CA1hsZx7YDMSmcEy6EU/edit?usp=sharing)
[[Data Collection Instruction]](https://swanky-sphere-ad1.notion.site/UMI-Data-Collection-Tutorial-4db1a1f0f2aa4a2e84d9742720428b4c?pvs=4)
[[SLAM repo]](https://github.com/cheng-chi/ORB_SLAM3)
[[SLAM docker]](https://hub.docker.com/r/chicheng/orb_slam3)

<img width="90%" src="assets/umi_teaser.png">

[Cheng Chi](http://cheng-chi.github.io/)<sup>1,2</sup>,
[Zhenjia Xu](https://www.zhenjiaxu.com/)<sup>1,2</sup>,
[Chuer Pan](https://chuerpan.com/)<sup>1</sup>,
[Eric Cousineau](https://www.eacousineau.com/)<sup>3</sup>,
[Benjamin Burchfiel](http://www.benburchfiel.com/)<sup>3</sup>,
[Siyuan Feng](https://www.cs.cmu.edu/~sfeng/)<sup>3</sup>,

[Russ Tedrake](https://groups.csail.mit.edu/locomotion/russt.html)<sup>3</sup>,
[Shuran Song](https://www.cs.columbia.edu/~shurans/)<sup>1,2</sup>

<sup>1</sup>Stanford University,
<sup>2</sup>Columbia University,
<sup>3</sup>Toyota Research Institute

## 🛠️ Installation
Only tested on Ubuntu 22.04

Install docker following the [official documentation](https://docs.docker.com/engine/install/ubuntu/) and finish [linux-postinstall](https://docs.docker.com/engine/install/linux-postinstall/).

Install system-level dependencies:
```console
$ sudo apt install -y libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf
```

We recommend [Miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge3) instead of the standard anaconda distribution for faster installation: 
```console
$ mamba env create -f conda_environment.yaml
```

Activate environment
```console
$ conda activate umi
(umi)$ 
```

## Camera Calibration

The SLAM pipeline requires camera-specific configuration files:

| File | Used by | Purpose |
|---|---|---|
| `gopro_intrinsics_2_7k.json` | Step 04 (ArUco detection) | Camera intrinsics (focal length, distortion) for tag pose estimation |
| `aruco_config.yaml` | Step 04 (ArUco detection) | ArUco marker dictionary and physical marker sizes |
| ORB_SLAM3 settings YAML | Steps 02 & 03 (SLAM) | Camera model, IMU params, and ORB feature settings for visual-inertial SLAM |
| `mask.json` | Steps 02, 03, 04 | Polygon definitions for masking mirrors, gripper body, and fingers |

These files are **camera-specific**. Using the wrong intrinsics will silently corrupt ArUco pose estimates (wrong z-depth), which breaks gripper width calibration and downstream training data.

### Included calibration data

| Directory | Camera | Resolution | Notes |
|---|---|---|---|
| `example/calibration/` | GoPro Hero 10 (MaxLens) | 2704 x 2028 | Ships with the upstream example data. **Do not modify.** |
| `gopro_cal_data/pipeline_calib/` | GoPro Hero 9 (custom) | 1352 x 1014 | Custom calibration for non-default camera setups |

The ORB_SLAM3 docker image also contains a **baked-in default** settings YAML for the GoPro 10 (`gopro10_maxlens_fisheye_setting_v1_720.yaml`). For other cameras, you must generate a custom YAML and pass it via `--slam_settings`.

### Using your own camera

If you are using a camera other than the original GoPro Hero 10 with MaxLens Mod, you need to:

1. **Calibrate your camera** using [OpenImuCameraCalibrator](https://github.com/urbste/OpenImuCameraCalibrator/) (see `Calibration_Tutorial.pdf` and `scripts/gen_orbslam3_yaml_from_openicc.py`).
2. **Create a calibration directory** containing `gopro_intrinsics_2_7k.json` and `aruco_config.yaml`.
3. **Generate an ORB_SLAM3 settings YAML** with your camera's intrinsics, IMU extrinsics, and noise parameters.
4. **Create a custom mask** (see below).
5. **Pass everything** to the pipeline:

```console
(umi)$ python run_slam_pipeline.py my_session \
    -c path/to/my_calibration_dir \
    -s path/to/my_slam_settings.yaml \
    -m path/to/my_mask.json
```

The pipeline prints which configuration files it is using at startup so you can always verify:

```
============================================================
Pipeline configuration
  Calibration dir  : /absolute/path/to/my_calibration_dir
  Camera intrinsics: .../gopro_intrinsics_2_7k.json
  ArUco config     : .../aruco_config.yaml
  SLAM settings    : /absolute/path/to/my_slam_settings.yaml
  Mask JSON        : /absolute/path/to/my_mask.json
============================================================
```

### Customizing the image mask

The pipeline masks out regions of the camera image (side mirrors, gripper body, finger area) to avoid confusing SLAM feature tracking and ArUco tag detection. By default it loads `umi/asset/mask.json`, which is tuned for the GoPro 10 + MaxLens setup. If your physical camera/gripper arrangement is different, you should create a custom `mask.json`.

**Step 1: Extract a reference frame** from your data for visual guidance:

```console
(umi)$ python scripts/extract_mask_frame.py -s my_session
Saved reference frame to: .../mask_reference_frame.png
  Resolution: 2704 x 2028  (width x height)
  "resolution": [2028, 2704]

(umi)$ python scripts/extract_mask_frame.py -s my_session --show_default_mask

(umi)$ python scripts/extract_mask_frame.py -s my_session --draw_masks
```

`--show_default_mask` writes overlay previews, while `--draw_masks` writes standalone binary mask PNGs (`*_slam_mask.png`, `*_training_mask.png`, `*_aruco_mask.png`) for quick evaluation.

**Step 2: Design your mask polygons.** Open the reference frame in an image viewer. Identify the pixel coordinates of the regions you need to mask. See `umi/asset/mask.json` for the default GoPro 10 template.

The `mask.json` format:

```json
{
    "mirror_mask_pts": [[x,y], ...],
    "gripper_mask_pts": [[x,y], ...],
    "finger_mask_pts": [[x,y], ...],
    "resolution": [height, width]
}
```

- `mirror_mask_pts` and `gripper_mask_pts` define **left-side** polygons in pixel coordinates. The right side is created automatically by mirroring the x-axis.
- `finger_mask_pts` defines the **full** polygon (not mirrored), typically a trapezoid covering the bottom of the frame.
- `resolution` is `[height, width]` matching the coordinate system of the points.
- All fields except `resolution` are optional; omitted fields fall back to the built-in GoPro 10 defaults.

**Step 3: Pass to the pipeline** with `-m path/to/my_mask.json`. If you omit `-m`, the pipeline uses `umi/asset/mask.json`.

## Running UMI SLAM pipeline

### Example data (GoPro 10)

Download example data:
```console
(umi)$ wget --recursive --no-parent --no-host-directories --cut-dirs=2 --relative --reject="index.html*" https://real.stanford.edu/umi/data/example_demo_session/
```

Run SLAM pipeline (uses the default `example/calibration/` intrinsics):
```console
(umi)$ python run_slam_pipeline.py example_demo_session
```

### Custom camera data

Run with explicit calibration directory, SLAM settings, and mask:
```console
(umi)$ python run_slam_pipeline.py my_session \
    -c gopro_cal_data/pipeline_calib \
    -s gopro_cal_data/gopro9_custom_1352x1014.yaml \
    -m path/to/my_mask.json
```

### Expected output

```
Found following cameras:
camera_serial
C3441328164125    5
Name: count, dtype: int64
Assigned camera_idx: right=0; left=1; non_gripper=2,3...
             camera_serial  gripper_hw_idx                                     example_vid
camera_idx                                                                                
0           C3441328164125               0  demo_C3441328164125_2024.01.10_10.57.34.882133
99% of raw data are used.
defaultdict(<function main.<locals>.<lambda> at 0x7f471feb2310>, {})
n_dropped_demos 0
```

For this dataset, 99% of the data are useable (successful SLAM), with 0 demonstrations dropped. If your dataset has a low SLAM success rate, double check if you carefully followed our [data collection instruction](https://swanky-sphere-ad1.notion.site/UMI-Data-Collection-Instruction-4db1a1f0f2aa4a2e84d9742720428b4c). 

Despite our significant effort on robustness improvement, OBR_SLAM3 is still the most fragile part of UMI pipeline. If you are an expert in SLAM, please consider contributing to our fork of [OBR_SLAM3](https://github.com/cheng-chi/ORB_SLAM3) which is specifically optimized for UMI workflow.

Generate dataset for training:
```console
(umi)$ python scripts_slam_pipeline/07_generate_replay_buffer.py -o example_demo_session/dataset.zarr.zip example_demo_session
```

## Training Diffusion Policy
Single-GPU training. Tested to work on RTX3090 24GB.
```console
(umi)$ python train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=example_demo_session/dataset.zarr.zip
```

Multi-GPU training.
```console
(umi)$ accelerate --num_processes <ngpus> train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=example_demo_session/dataset.zarr.zip
```

Downloading in-the-wild cup arrangement dataset (processed).
```console
(umi)$ wget https://real.stanford.edu/umi/data/zarr_datasets/cup_in_the_wild.zarr.zip
```

Multi-GPU training.
```console
(umi)$ accelerate --num_processes <ngpus> train.py --config-name=train_diffusion_unet_timm_umi_workspace task.dataset_path=cup_in_the_wild.zarr.zip
```

## 🦾 Real-world Deployment
In this section, we will demonstrate our real-world deployment/evaluation system with the cup arrangement policy. While this policy setup only requires a single arm and camera, the our system supports up to 2 arms and unlimited number of cameras.

### ⚙️ Hardware Setup
1. Build deployment hardware according to our [Hardware Guide](https://docs.google.com/document/d/1TPYwV9sNVPAi0ZlAupDMkXZ4CA1hsZx7YDMSmcEy6EU).
2. Setup UR5 with teach pendant:
    * Obtain IP address and update [eval_robots_config.yaml](example/eval_robots_config.yaml)/robots/robot_ip.
    * In Installation > Payload
        * Set mass to 1.81 kg
        * Set center of gravity to (2, -6, 37)mm, CX/CY/CZ.
    * TCP will be set automatically by the eval script.
    * On UR5e, switch control mode to remote.

    If you are using Franka, follow this [instruction](franka_instruction.md).
3. Setup WSG50 gripper with web interface:
    * Obtain IP address and update [eval_robots_config.yaml](example/eval_robots_config.yaml)/grippers/gripper_ip.
    * In Settings > Command Interface
        * Disable "Use text based Interface"
        * Enable CRC
    * In Scripting > File Manager
        * Upload [umi/real_world/cmd_measure.lua](umi/real_world/cmd_measure.lua)
    * In Settings > System
        * Enable Startup Script
        * Select `/user/cmd_measure.lua` you just uploaded.
4. Setup GoPro:
    * Install GoPro Labs [firmware](https://gopro.com/en/us/info/gopro-labs).
    * Set date and time.
    * Scan the following QR code for clean HDMI output 
    <br><img width="50%" src="assets/QR-MHDMI1mV0r27Tp60fWe0hS0sLcFg1dV.png">
5. Setup [3Dconnexion SpaceMouse](https://www.amazon.com/3Dconnexion-SpaceMouse-Wireless-universal-receiver/dp/B079V367MM):
    * Install libspnav `sudo apt install libspnav-dev spacenavd`
    * Start spnavd `sudo systemctl start spacenavd`

### 🤗 Reproducing the Cup Arrangement Policy ☕
Our in-the-wild cup arragement policy is trained with the distribution of ["espresso cup with saucer"](https://www.amazon.com/s?k=espresso+cup+with+saucer) on Amazon across 30 different locations around Stanford. We created a [Amazon shopping list](https://www.amazon.com/hz/wishlist/ls/Q0T8U2N5U3IU?ref_=wl_share) for all cups used for training. We published the processed [Zarr dataset and](https://real.stanford.edu/umi/data/zarr_datasets) pre-trained [checkpoint](https://real.stanford.edu/umi/data/pretrained_models/) (finetuned CLIP ViT-L backbone).

<img width="90%" src="assets/umi_cup.gif">

Download pre-trained checkpoint.
```console
(umi)$ wget https://real.stanford.edu/umi/data/pretrained_models/cup_wild_vit_l_1img.ckpt
```

Grant permission to the HDMI capture card.
```console
(umi)$ sudo chmod -R 777 /dev/bus/usb
```

Launch eval script.
```console
(umi)$ python eval_real.py --robot_config=example/eval_robots_config.yaml -i cup_wild_vit_l.ckpt -o data/eval_cup_wild_example
```
After the script started, use your spacemouse to control the robot and the gripper (spacemouse buttons). Press `C` to start the policy. Press `S` to stop.

If everything are setup correctly, your robot should be able to rotate the cup and placing it onto the saucer, anywhere 🎉

Known issue ⚠️: The policy doesn't work well under direct sunlight, since the dataset was collected during a rainiy week at Stanford.

### 🤗 Reproducing Policies on ARX X5 Robot Arms
Please follow [umi-on-legs](https://github.com/real-stanford/umi-on-legs) for hardware modification and [umi-arx](https://github.com/real-stanford/umi-arx) for detailed policy deployment instructions. 

<img width="90%" src="assets/umi_cup_arx.gif">

## 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.

## 🙏 Acknowledgement
* Our GoPro SLAM pipeline is adapted from [Steffen Urban](https://github.com/urbste)'s [fork](https://github.com/urbste/ORB_SLAM3) of [OBR_SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3).
* We used [Steffen Urban](https://github.com/urbste)'s [OpenImuCameraCalibrator](https://github.com/urbste/OpenImuCameraCalibrator/) for camera and IMU calibration.
* The UMI gripper's core mechanism is adpated from [Push/Pull Gripper](https://www.thingiverse.com/thing:2204113) by [John Mulac](https://www.thingiverse.com/3dprintingworld/designs).
* UMI's soft finger is adapted from [Alex Alspach](http://alexalspach.com/)'s original design at TRI.
