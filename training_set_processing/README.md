# Prepare your own dataset

Our implementation of the training set is using the implementation excellent implementation of [The Vector Heat Method](https://nmwsharp.com/research/vector-heat-method/)

## Prerequisites
Compile the external code that computes logmaps:
``` bash
cd logmap
mkdir build
cd build
cmake ..
make
```

## Run

* Copy you shapes in the `data` directory
* prepare_dataset.py


The produced data files are numbered from 0 to the number of files. However, the correspondence is saved in valid_files_list.txt
Our full unprocessed dataset is available online: [Download dataset](https://nuage.lix.polytechnique.fr/index.php/s/GZyqbDmtSP889zt).
