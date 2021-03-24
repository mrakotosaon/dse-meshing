# [Learning Delaunay Surface Elements for Mesh Reconstruction (DSE meshing)](http://www.lix.polytechnique.fr/Labo/Marie-Julie.RAKOTOSAONA/dse_meshing.html)
This is our implementation of the paper ["Learning Delaunay Surface Elements for Mesh Reconstruction"](https://arxiv.org/abs/2012.01203) at CVPR 2021 (oral), a method for mesh recontruction from a point cloud.


![DSE_meshing](img/dse_meshing_teaser.png "DSE meshing")


This code was written by [Marie-Julie Rakotosaona](http://www.lix.polytechnique.fr/Labo/Marie-Julie.RAKOTOSAONA/).

The triangle selection code is based on  on [Meshing-Point-Clouds-with-IER](https://github.com/Colin97/Point2Mesh) (with a few smaller modifications) that also uses code from the project [annoy](https://github.com/spotify/annoy). Our network code is based on [PointNet](https://github.com/charlesq34/pointnet).

## Prerequisites
* CUDA and CuDNN (changing the code to run on CPU should require few changes)
* Python 3.6
* Tensorflow 1.5

## Setup
Install required python packages, if they are not already installed:
``` bash
pip install numpy
pip install scipy
pip install trimesh
```


Clone this repository:
``` bash
git clone https://github.com/mrakotosaon/dse-meshing.git
cd dse-meshing
```

Setup the triangle selection step:
``` bash
git submodule update --init --recursive
cd triangle_selection/postprocess
mkdir build
cd build
cmake ..
make
```



 ## Data


- **Training set:** We store our training data in .tfrecords files. The files contain elements of size N_NEIGHBORSx5 where the first 3 columns contain the 3D coordinates and the last two columns contain the ground truth logmap 2D coordinates.
- **Pre-trained models** on the famousthingi dataset.
- **Testset** from famousthingi dataset.


Our data can be downloaded directly here:
- training data: https://nuage.lix.polytechnique.fr/index.php/s/gmnGHjNq7WKipRA
- pretrained models: https://nuage.lix.polytechnique.fr/index.php/s/FTCyp5WHg7Z68EM
- testing data: https://nuage.lix.polytechnique.fr/index.php/s/3ZcFtqKm6Z27ZJ6

To download our data from the code:
- Download pretrained models:
  ``` bash
  cd data
  python download_data.py --task models
  ```

- Download training set:
  ``` bash
  cd data
  python download_data.py --task training
  ```

- Download testing set:
  ``` bash
  cd data
  python download_data.py --task testing
  ```

## Training
To train the classifier network on the provided dataset:
``` bash
cd train_logmap
python train_classifier.py
```

To train the logmap estimation network on the provided dataset:
``` bash
cd train_logmap
python train_logmap_network.py
```

- The trained models are saved in `log/log_famousthingi_classifier` and `log/log_famousthingi_logmap` by default. Paths for the training set, output and training parameters can be changed directly in the code.
- Training curves are generated during training and can be viewed using `tensorboard`.



## Testing
Our testing pipeline has 3 main steps:
1.  **Logmap estimation:** we locally predict the logmap that contains neighboring points at each point using the trained networks.
2. **Logmap alignment:** we locally align the log maps to one another to ensure better consistency.
3. **Triangle selection:** we select the produced triangles to generate an almost manifold mesh.

We provide one example point cloud in `data/test_data`. For easily evaluating your pointclouds or the pointclouds from the testset, move the .xyz files there.

To run all steps on the point clouds in data/test_data directory:
``` bash
 ./run.sh
```
The produced meshes can be found in `data/test_data/select` in the format: `final_mesh_(SHAPE_NAME).ply`. For numerical precision reasons we suggest to use point clouds with a bounding box diagonal larger than 1. By default our code evaluates the .xyz point clouds in `data/test_data` please adapt the paths in the code if you wish to evaluate  point clouds at different locations.
### 1. Logmap estimation

To run only the logmap estimation networks on the point clouds in data/test_data directory:
``` bash
cd logmap_estimation
python eval_network.py
```

### 2. Logmap alignment

To only align the logmap patches from step 1 and compute the appearance frequency for step 3:
``` bash
cd logmap_alignment
python align_patches.py
python eval_align_meshes.py
```

### 3. Triangle selection

To apply triangle selection on the output from step 2:
``` bash
cd triangle_selection
python select.py
```


## Citation
If you use our work, please cite our paper.
```
@article{rakotosaona2020learning,
  title={Learning Delaunay Surface Elements for Mesh Reconstruction},
  author={Rakotosaona, Marie-Julie and Guerrero, Paul and Aigerman, Noam and Mitra, Niloy and Ovsjanikov, Maks},
  journal={arXiv e-prints},
  pages={arXiv--2012},
  year={2020}
}
```

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). For any commercial uses or derivatives, please contact us.
