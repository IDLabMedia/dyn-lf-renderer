# Preprocessor

The preprocessor transforms the multi-view input videos to an intermediate representation that can be used by the renderer.

## Getting started
### Installing the preprocessor
#### Virtual env
First create a virtual environment (skip this step if you want to install globally).
```bash
python -m venv .venv
```
and activate it:

Linux/macOs:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

#### Install package
Now install the package:
```bash
pip install .
```
If you have an nvidia GPU, install using:
```bash
pip install .[nvidia]
```
this allows jax to run on your gpu, instead of on your cpu.

#### Usage
Show all command line argument options with:
```bash
rtdlf-preprocessor -h
```
An example setup could be:
```bash
rtdlf-preprocessor -v voxel_cloud centers -f yuv
```
This will generate a voxel aligned point cloud with inpainting data, center vertex points for each cameras and the yuv colour files. These are all needed by the renderer.

## Intput data
The input of the preprocessor are a depth and color video for each camera that captured the scene.
The camera intrinsics and extrinsics should also be specified.

### Camera information
The camera information (intrinsics and extrinsics) should be described in a json file in the dataset folder. The path to this folder must be specified as a[CLI argument](#usage).
The file should have the same name as the parent directory, but ending with `.json`
The json should have the following structure:
```json
{
  "Resolution": [<px-width>, <px-height>],
  "Focal": [<focal-x>, <focal-y>],
  "Principal_point": [<pp-x>, <pp-y>],
  "Depth_range": [<near>, <far>],
  "cameras": [
    {
      "NameColor": "<name-of-color-video>",
      "NameDepth": "<name-of-depth-video>",
      "model": <model-matrix>
    },
    {...},
    ...
  ]
}
```
#### Notes on the format
- The `Resolution`, `Focal`, `Pricipal_point` and `Depth_range` can be specified per camera, by adding those fields to the camera object. This will override the default values. It is not needed to set a default value if all cameras have a value specified.
- The `NameColor` and `NameDepth` are the names of the videos taken by this camera. These files should be present in the same directory as the camera.
- The model matrix of a camera is a 4x4 matrix, made up of the rotation matrix and the position of that camera in the global axial system of the scene. This matrix represents the transform from camera space to world space.

```math
\text{Model Matrix} = 
\begin{matrix}
\text{R}_{11} & \text{R}_{12} & \text{R}_{13} & \text{C}_x \\
\text{R}_{21} & \text{R}_{22} & \text{R}_{23} & \text{C}_y \\
\text{R}_{31} & \text{R}_{32} & \text{R}_{33} & \text{C}_z \\
0 & 0 & 0 & 1
\end{matrix}
```
### Colour videos
The videos should all have an equal amount of frames, and the same resolution (the same holds for the depth videos).
The colour videos should be RGB, and in a valid format that is readable by opencv.
The file names should be the same as specified in the json file.

### Depth videos
The videos should all have an equal amount of frames, and the same resolution (the same holds for the colour videos).
The depth videos must be raw yuv videos in the `yuv420p10le` format. `ffmpeg` can be used to generate yuv videos from your video fromat:

```bash
ffmpeg -i <input.ext> -pix_fmt yuv420p10le -f rawvideo output.yuv
```
The [`decompress_depth.sh`](./decompress_depth.sh) script transforms all `*_depth.mp4` files to yuv.
If the names of the depth files end in `.mp4` in the json file, their `.yuv` counterpart will tried to be loaded instead.
Otherwise should the name of the depth video match the name in the json file.

### Folder layout
An example intput folder layout is specified below.

```txt
Frog
├── Frog.json
├── v01_depth.mp4
├── v01_texture.yuv
├── v02_depth.mp4
├── v02_texture.yuv
├── v03_depth.mp4
├── v03_texture.yuv
├── v04_depth.mp4
├── v04_texture.yuv
├── v05_depth.mp4
└── v05_texture.yuv
```
