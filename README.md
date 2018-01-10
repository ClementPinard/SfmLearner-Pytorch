# SfMLearner Pytorch version
This codebase implements the system described in the paper:

Unsupervised Learning of Depth and Ego-Motion from Video

[Tinghui Zhou](https://people.eecs.berkeley.edu/~tinghuiz/), [Matthew Brown](http://matthewalunbrown.com/research/research.html), [Noah Snavely](http://www.cs.cornell.edu/~snavely/), [David G. Lowe](http://www.cs.ubc.ca/~lowe/home.html)

In CVPR 2017 (**Oral**).

See the [project webpage](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/) for more details. 

Original Author : Tinghui Zhou (tinghuiz@berkeley.edu)
Pytorch implementation : Clément Pinard (clement.pinard@ensta-paristech.fr)

![sample_results](misc/cityscapes_sample_results.gif)

## Preamble
This codebase was developed and tested with Pytorch 0.2, CUDA 8.0 and Ubuntu 16.04. Original code was developped in tensorflow, you can access it [here](https://github.com/tinghuiz/SfMLearner)

## Prerequisite

```bash
[sudo] pip3 install -r requirements.txt
```

or install manually the following packages :

```
pytorch 0.3
scipy
argparse
tensorboard-pytorch
tensorboard
blessings
progressbar2
path.py
```

It is also advised to have python3 bindings for opencv for tensorboard visualizations

### What has been done (for the moment)

* Training has been tested on KITTI and CityScapes. Convergence is reached, although with a different set of hyperparameters.
* Dataset preparation has been largely improved, and now stores image sequences in folders, making sure that movement is each time big enough between each frame
* That way, training is now significantly faster, running at ~0.14sec per step vs ~0.2s per steps initially (on a single GTX980Ti)
* In addition you don't need to prepare data for a particular sequence length anymore as stacking is made on the fly.
* You can still choose the former stacked frames dataset format.
* Convergence is now almost as good as original paper with same hyper parameters
* You can know compare with groud truth for your validation set. It is still possible to validate without, but you know can see that minimizing photometric error is not equivalent to optimizing depth map.

### Still needed to do

* Pose evaluation code

## Preparing training data
Preparation is roughly the same command as in the original code.

For [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), first download the dataset using this [script](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website, and then run the following command. The `--with-gt` option will save resized copies of groudtruth to help you setting hyper parameters.
```bash
python3 data/prepare_train_data.py /path/to/raw/kitti/dataset/ --dataset-format 'kitti' --dump-root /path/to/resulting/formatted/data/ --width 416 --height 128 --num-threads 4 [--static-frames /path/to/static_frames.txt] [--with-gt]
```


For [Cityscapes](https://www.cityscapes-dataset.com/), download the following packages: 1) `leftImg8bit_sequence_trainvaltest.zip`, 2) `camera_trainvaltest.zip`. You will probably need to contact the administrators to be able to get it. Then run the following command
```bash
python3 data/prepare_train_data.py /path/to/cityscapes/dataset/ --dataset-format 'cityscapes' --dump-root /path/to/resulting/formatted/data/ --width 416 --height 171 --num-threads 4
```
Notice that for Cityscapes the `img_height` is set to 171 because we crop out the bottom part of the image that contains the car logo, and the resulting image will have height 128.

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
python3 test_disp.py --pretrained-dispnet /path/to/dispnet --pretrained-posent /path/to/posenet --dataset-dir /path/to/KITTI_raw --dataset-list /path/to/test_files_list
```

Test file list is available in kitti eval folder. To get fair comparison with [Original paper evaluation code](https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/eval_depth.py), don't specify a posenet. However, if you do,  it will be used to solve the scale factor ambiguity, the only ground truth used to get it will be vehicle speed which is far more acceptable for real conditions quality measurement, but you will obviously get worse results.

## Other Implementations

[TensorFlow](https://github.com/tinghuiz/SfMLearner) by tinghuiz (original code, and paper author)
