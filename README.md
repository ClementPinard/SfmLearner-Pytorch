# 16726 - Learning-based Image synthesis - Final Project 

### SemSyn: Semantics-driven View-Synthesis 

This code contains the implementation for the final project of 16-726.

<div align="center">
    <img src="assets/sample.gif"/>
</div>

The backbone of the code is based on this [repo](https://github.com/ClementPinard/SfmLearner-Pytorch) 
which is implements the following paper: 

**Unsupervised Learning of Depth and Ego-Motion from Video** by
[Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), 
[Matthew Brown](http://matthewalunbrown.com/research/research.html), 
[Noah Snavely](http://www.cs.cornell.edu/~snavely/), 
[David G. Lowe](http://www.cs.ubc.ca/~lowe/home.html)

For our project, we:
- Extend the framework to perform view-synthesis and Structure-from-Motion on indoor environments.
- Provide instructions and utility code needed to generate the indoor dataset.
- Extend it to support semantic labels as inputs for predicting depth information.
- Extend the supported loss functions.

## Requirements and setup
This code was tested with Pytorch 1.11.0, CUDA 11.4 and Ubuntu 18.04. We 
followed the setup below: 

```bash
conda create -n sfm python=3.7
conda activate sfm
pip install -r requirements.txt
```

## Preparing the dataset 

We used the Matterport3D ([MP3D](https://niessner.github.io/Matterport/)) dataset for 
our project and the [Habitat](https://aihabitat.org) simulation environment to 
generate egocentric trajectories for training, validation and testing. 

<div align="center">
    <img src="assets/rgb.gif"/>
    <img src="assets/depth.gif"/>
    <img src="assets/semantics.gif"/>
</div>

### How to get the Matterport3D dataset?

##### Scenes:
We use the MP3D's scene reconstructions. The official Matterport3D download script 
(`download_mp.py`) can be accessed by following the instructions [here](https://niessner.github.io/Matterport/). 
The scene data can then be downloaded:
```bash
# requires running with python 2.7
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

We extracted it such that it has the form `data/scene_datasets/mp3d/{scene}/{scene}.glb`.
There should be 90 scenes.

##### Trajectories:
We use the Vision-Language Navigation in Continuous Environments ([VLN-CE](https://jacobkrantz.github.io/vlnce/)) 
dataset for getting the ego-centric trajectories. We used the ```R2R_VLNCE_v1-3_preprocessed``` 
version of the dataset from their [repository](https://github.com/jacobkrantz/VLN-CE).
We extracted it such that it has the form `data/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz`.

##### Simulator:
We used the [Habitat-Sim](git@github.com:facebookresearch/habitat-sim.git) and 
[Habitat-Lab](git@github.com:facebookresearch/habitat-lab.git) to prepare de dataset.
To install them, run:

```bash
# this is habitat-sim
conda install -c aihabitat -c conda-forge habitat-sim=0.1.7 headless

# this is for habitat-lab
git clone --branch v0.1.7 git@github.com:facebookresearch/habitat-lab.git
cd habitat-lab
python -m pip install -r requirements.txt
python setup.py develop --all
```

##### The actual data we used:
Finally, in ```data/habitat_extension``` we provide the code we used to generate
the data we used in this project. Run it as:
```bash
python data/habitat_extension/run.py --exp-config data/habitat_extension/mp3d.yaml
```
The code should generate color, depth and semantic images, and pose information 
for each trajectory, as shown in the gif above. 

##### TODO: Post-processing steps:

Additional post-processing was performed (does not require simulator):
- Switching color channels
- Converting the semantic images to labels 
- Adding intrinsic parameters to the dataset
- Statistics 

# TODO: everything below needs to get modified

## Training
Once the data are formatted following the above instructions, you should be able to train the model by running the following command
```bash
python3 train.py /path/to/the/formatted/data/ -b4 -m0.2 -s0.1 --epoch-size 3000 --sequence-length 3 --log-output [--with-gt]
```
You can then start a `tensorboard` session in this folder by
```bash
tensorboard --logdir=checkpoints/
```
and visualize the training progress by opening [https://localhost:6006](https://localhost:6006) on your browser. If everything is set up properly, you should start seeing reasonable depth prediction after ~30K iterations when training on KITTI.

## Evaluation

Disparity map generation can be done with `run_inference.py`
```bash
python3 run_inference.py --pretrained /path/to/dispnet --dataset-dir /path/pictures/dir --output-dir /path/to/output/dir
```
Will run inference on all pictures inside `dataset-dir` and save a jpg of disparity (or depth) to `output-dir` for each one see script help (`-h`) for more options.

Disparity evaluation is avalaible
```bash
python3 test_disp.py --pretrained-dispnet /path/to/dispnet --pretrained-posenet /path/to/posenet --dataset-dir /path/to/KITTI_raw --dataset-list /path/to/test_files_list
```

Test file list is available in kitti eval folder. To get fair comparison with [Original paper evaluation code](https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/eval_depth.py), don't specify a posenet. However, if you do,  it will be used to solve the scale factor ambiguity, the only ground truth used to get it will be vehicle speed which is far more acceptable for real conditions quality measurement, but you will obviously get worse results.

Pose evaluation is also available on [Odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php). Be sure to download both color images and pose !

```bash
python3 test_pose.py /path/to/posenet --dataset-dir /path/to/KITIT_odometry --sequences [09]
```

**ATE** (*Absolute Trajectory Error*) is computed as long as **RE** for rotation (*Rotation Error*). **RE** between `R1` and `R2` is defined as the angle of `R1*R2^-1` when converted to axis/angle. It corresponds to `RE = arccos( (trace(R1 @ R2^-1) - 1) / 2)`.
While **ATE** is often said to be enough to trajectory estimation, **RE** seems important here as sequences are only `seq_length` frames long.

## Pretrained Nets

[Avalaible here](https://drive.google.com/drive/folders/1H1AFqSS8wr_YzwG2xWwAQHTfXN5Moxmx)

Arguments used :

```bash
python3 train.py /path/to/the/formatted/data/ -b4 -m0 -s2.0 --epoch-size 1000 --sequence-length 5 --log-output --with-gt
```

### Depth Results

| Abs Rel | Sq Rel | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
|---------|--------|-------|-----------|-------|-------|-------|
| 0.181   | 1.341  | 6.236 | 0.262     | 0.733 | 0.901 | 0.964 | 

### Pose Results

5-frames snippets used

|    | Seq. 09              | Seq. 10              |
|----|----------------------|----------------------|
|ATE | 0.0179 (std. 0.0110) | 0.0141 (std. 0.0115) |
|RE  | 0.0018 (std. 0.0009) | 0.0018 (std. 0.0011) | 


## Discussion

Here I try to link the issues that I think raised interesting questions about scale factor, pose inference, and training hyperparameters

 - [Issue 48](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/48) : Why is target frame at the center of the sequence ?
 - [Issue 39](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/39) : Getting pose vector without the scale factor uncertainty
 - [Issue 46](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/46) : Is Interpolated groundtruth better than sparse groundtruth ?
 - [Issue 45](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/45) : How come the inverse warp is absolute and pose and depth are only relative ?
 - [Issue 32](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/32) : Discussion about validation set, and optimal batch size
 - [Issue 25](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/25) : Why filter out static frames ?
 - [Issue 24](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/24) : Filtering pixels out of the photometric loss
 - [Issue 60](https://github.com/ClementPinard/SfmLearner-Pytorch/issues/60) : Inverse warp is only one way !

## Other Implementations

[TensorFlow](https://github.com/tinghuiz/SfMLearner) by tinghuiz (original code, and paper author)
