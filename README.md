#Simultaneous Safe Screening of Features and Samples in Doubly Sparse Modeling (ICML'16)

##Enviromental Requirement
* gcc version *> 4.8.0*
* cmake version *> 2.8.12*

##About
Source code for Simultaneous Safe Screening of Features and Samples in Doubly Sparse Modeling.
We wrote the code in C++ along with
[Eigen3.3-alpha1](http://eigen.tuxfamily.org/index.php?title=Main_Page) library for some numerical computations.

##How to Compile
```
cd s3fs
cmake .
make
```

## Usage

###We support [LIBSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) data fortmat only ( [LIBSVM datasets](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/) ).

###Elastic net + smoothed hinge loss (model selection task) with SPDC
- non-screeing:  `./test/elastic_smooth_module -s 14 -e 1e-9 [dataset_filename] `
- simultaneous safe screeing: `./test/elastic_smooth_module -s 15 -e 1e-9 - d 1 [dataset_filename] `
- safe feature screeing:  `./test/elastic_smooth_module -s 16 -e 1e-9 -d 1 [dataset_filename] `
- safe sample screeing:  `./test/elastic_smooth_module -s 17 -e 1e-9 - d 1 [dataset_filename] `

###Elastic net + smoothed epsilon-insensitive loss (model selection task) with SPDC
- non-screeing:  `./test/elastic_soft_module -s 14 -e 1e-9 [dataset_filename] `
- simultaneous safe screeing: `./test/elastic_soft_module -s 15 -e 1e-9 - d 1 [dataset_filename] `
- safe feature screeing:  `./test/elastic_soft_module -s 16 -e 1e-9 -d 1 [dataset_filename] `
- safe sample screeing:  `./test/elastic_soft_module -s 17 -e 1e-9 - d 1 [dataset_filename] `

