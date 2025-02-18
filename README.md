<h1 align="center">
Sim2Real Transfer for Vision-Based Grasp Verification
</h1>

<div align="center">
<h3>
<a href="https://github.com/pauamargant">Pau Amargant</a>,
<a href="https://github.com/hoenigpeter">Peter Hönig</a>,
<a href="http://github.com/v4r-tuwien">Markus Vincze</a>,
<br>
<br>
</h3>
</div>

<p align="center">
  <img src="images/pos_01.png" width="30%" />
  <img src="images/pos_02.png" width="30%" />
  <img src="images/pos_03.png" width="30%" />
  <br>
  <img src="images/neg_01.jpg" width="30%" />
  <img src="images/neg_02.jpg" width="30%" />
  <img src="images/neg_03.jpg" width="30%" />
</p>

# HSR-GraspSynth Dataset
This repository contains the script to generate the HSR-GraspSynth dataset. The dataset is composed of synthetic images of the HSR robot grasping objects in a table. The dataset is generated using BlenderProc and the URDF model of the HSR robot.

To generate the dataset, use the rendering script provided in `blenderproc_scripts/`. You will need to place the following datasets and files and define the paths in the `config` file:
- [BlenderProc](https://github.com/DLR-RM/BlenderProc.git)
- URDF [model](https://github.com/ToyotaResearchInstitute/hsr_description) of the HSR robot
- [ShapeNet](https://shapenet.org/) objects
- [CC0 Textures](https://cc0textures.com/) 

## Augmentations
During the training process of the GraspCheckNet model proposed in the paper, the dataset was augmented using the following transformations from the [Albumentations](albumenatations.ai) library:
 

| **Augmentation**            | **Parameters** |
|-----------------------------|---------------|
| Perspective Transform       | Scale: [0.05, 0.1], p = 0.5 |
| Random Crop and Pad         | Percent: [-0.05, 0.1], p = 0.5 |
| Affine Transform            | Scale: [1.0, 1.2], p = 0.5 |
| Coarse Dropout              | Max Holes: 8, Max Height: 10, Max Width: 10, Min Height: 5, Min Width: 5, p = 0.5 |
| Gaussian Blur               | Blur Limit: [3, 7], p = 0.7 |
| Color Jitter               | Brightness: [0.8, 1.2], Contrast: [0.2, 1.2], Saturation: [0.2, 0.8], Hue: [-0.1, 0.1], p = 1 |
| Random Brightness-Contrast  | Brightness Limit: [-0.25, 0.25], Contrast Limit: [-0.2, 0.2], p = 0.5 |
| Gaussian Noise              | Mean: 0, Std Range: [0.05, 0.15], p = 0.5 |
| Optical Distortion          | Distort Limit: 0.05, Shift Limit: 0.05, p = 0.5 |
| Image Compression           | JPEG Compression |

