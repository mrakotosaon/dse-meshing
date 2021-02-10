# [Learning Delaunay Surface Elements for Mesh Reconstruction (DSE meshing)](http://www.lix.polytechnique.fr/Labo/Marie-Julie.RAKOTOSAONA/dse_meshing.html)
This is our implementation of "Learning Delaunay Surface Elements for Mesh Reconstruction", a method for mesh recontruction from a point cloud.


![DSE_meshing](img/dse_meshing_teaser.png "DSE meshing")


This code was written by [Marie-Julie Rakotosaona](http://www.lix.polytechnique.fr/Labo/Marie-Julie.RAKOTOSAONA/).

The triangle selection code is based on  on [Meshing-Point-Clouds-with-IER](https://github.com/Colin97/Point2Mesh) (with a few smaller modifications) that also uses code from the project [annoy](https://github.com/spotify/annoy).

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
git clone https://github.com/mrakotosaon/pointcleannet.git
cd pointcleannet
```


Download datasets:
``` bash
cd data
python download_data.py --task denoising
python download_data.py --task outliers_removal
```


Download pretrained models:
``` bash
cd models
python download_models.py --task denoising
python download_models.py --task outliers_removal
```

 ## Data

Our data can be found here: https://nuage.lix.polytechnique.fr/index.php/s/xSRrTNmtgqgeLGa .

It contains the following files:
- Dataset for denoising
- Training set and test set for outliers removal
- Pre-trained models for denoising and outliers removal

In the datasets the input and ground truth point clouds are stored in different files with the same name but with different extensions.
- For denoising: `.xyz` for input noisy point clouds, `.clean_xyz` for the ground truth.
- For outliers removal: `.xyz` for input point clouds with outliers, `.outliers` for the labels.



## Training
To classify outliers using default settings:
``` bash
cd outliers_removal
mkdir results
python eval_pcpnet.py
```

## Testing
Our testing pipeline has 4 main steps:
- **Logmap estimation:** we locally predict the logmap that contains neighboring points at each point using the trained networks.
- **Logmap alignment:** we locally align the log maps to one another to ensure better consistency.
- **Triangle selection:** we select the produced triangles to generate an almost manifold mesh.

### Logmap estimation
### Logmap alignment
### Triangle selection
``` bash
cd noise_removal
mkdir results
./run.sh
```
(the input shapes and number of iterations are specified in run.sh file)


## Training
To train PCPNet with the default settings:
``` bash
python train_pcpnet.py
```

## Citation
If you use our work, please cite our paper.
```
@inproceedings{rakotosaona2021dsemeshing,
  title={Learning Delaunay Surface Elements for Mesh Reconstruction},
  author={Rakotosaona, Marie-Julie and Guerrero, Paul and Aigerman, Noam and Mitra, Niloy J and Ovsjanikov, Maks},
  year={2021},
}
```

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). For any commercial uses or derivatives, please contact us.
