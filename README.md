#Simultaneous Safe Screening of Features and Samples in Doubly Sparse Modeling (ICML'16)

<img src="fig/sss_illust2.pdf" width="920px">


##Abstract

The problem of learning a sparse model is conceptually interpreted as the process of identifying active features/samples and then optimizing the model over them. Recently introduced safe screening allows us to identify a part of non-active features/samples. So far, safe screening has been individually studied either for feature screening or for sample screening. In this paper, we introduce a new approach for safely screening features and samples simultaneously by alternatively iterating feature and sample screening steps. A significant advantage of considering them simultaneously rather than individually is that they have a synergy effect in the sense that the results of the previous safe feature screening can be exploited for improving the next safe sample screening performances, and vice-versa. We first theoretically investigate the synergy effect, and then illustrate the practical advantage through intensive numerical experiments for problems with large numbers of features and samples.

##Result

<img src="fig/0205_rate2_real-sim.pdf" width="920px">


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

