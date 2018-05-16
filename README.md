# PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition
**[PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition](https://arxiv.org/abs/1804.03492)** CVPR 2018, Salt Lake City, USA

Mikaela Angelina Uy and Gim Hee Lee

National University of Singapore

## Introduction
The PointNetVLAD is a deep network that addresses the problem of large-scale place recognition through point cloud based retrieval. The arXiv version of PointNetVLAD can be found [here](https://arxiv.org/abs/1804.03492).
```
@article{uy2018pointnetvlad,
      title={PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition},
      author={Uy, Mikaela Angelina and Lee, Gim Hee},
      journal={arXiv preprint arXiv:1804.03492},
      year={2018}
}
```
## Benchmark Datasets
The benchmark datasets introdruced in this work can be downloaded [here](https://drive.google.com/open?id=1Wn1Lvvk0oAkwOUwR0R6apbrekdXAUg7D).
* All submaps are in binary file format
* Ground truth GPS coordinate of the submaps are found in the corresponding csv files for each run
* Filename of the submaps are their timestamps which is consistent with the timestamps in the csv files
* Use CSV files to define positive and negative point clouds
* All submaps are preprocessed with the road removed and downsampled to 4096 points

### Oxford Dataset
* 45 sets in total of full and partial runs
* Used both full and partial runs for training but only used full runs for testing/inference
* Training submaps are found in the folder "pointcloud_20m_10overlap/" and its corresponding csv file is "pointcloud_locations_20m_10overlap.csv"
* Training submaps are not mutually disjoint per run
* Each training submap ~20m of car trajectory and subsequent submaps are ~10m apart
* Test/Inference submaps found in the folder "pointcloud_20m/" and its corresponding csv file is "pointcloud_locations_20m.csv"
* Test/Inference submaps are mutually disjoint

### Inhouse Datasets

* Each inhouse dataset has 5 runs
* Training submaps are found in the folder "pointcloud_25m_10/" and its corresponding csv file is "pointcloud_centroids_10.csv"
* Test/Infenrence submaps are found in the folder "pointcloud_25m_25/" and its corresponding csv file is "pointcloud_centroids_25.csv"
* Training submaps are not mutually disjoint per run but test submaps are

## Project Code

### Pre-requisites
* Python
* CUDA
* Tensorflow 
* Scipy
* Pandas
* Sklearn

Code was tested using Python 3 on Tensorflow 1.4.0 with CUDA 8.0

```
sudo apt-get install python3-pip python3-dev python-virtualenv
virtualenv --system-site-packages -p python3 ~/tensorflow
source ~/tensorflow/bin/activate
easy_install -U pip
pip3 install --upgrade tensorflow-gpu==1.4.0
pip install scipy, pandas, sklearn
```
### Dataset set-up
Download the zip file of the benchmark datasets found [here](https://drive.google.com/open?id=1rflmyfZ1v9cGGH0RL4qXRrKhg-8A-U9q). Extract the folder on the same directory as the project code. Thus, on that directory you must have two folders: 1) benchmark_datasets/ and 2) pointnetvlad/

### Generate pickle files
We store the positive and negative point clouds to each anchor on pickle files that are used in our training and evaluation codes. The files only need to be generated once. The generation of these files may take a few minutes.

```
# For training tuples in our baseline network
python generate_training_tuples_baseline.py

# For training tuples in our refined network
python generate_training_tuples_refine.py

# For network evaluation
python generate_test_sets.py
```

### Model Training and Evaluation
To train our network, run the following command:
```
python train_pointnetvlad.py
```
To evaluate the model, run the following command:
```
python evaluate.py
```

## Pre-trained Models
The pre-trained models for both the baseline and refined networks can be downloaded [here](https://drive.google.com/open?id=1wYsJmfd2yfbK9DHjFHwEeU1a_x35od61)
