# SkeletonActionRecognition
A collection of skeleton-based human action recognition methods using PyTorch

## Contents
1. [Installation](#installation)
2. [Usage](#usage)
   1. [Examples](#examples)
   2. [Submodules](#submodules)
3. [Testing](#testing)

## Installation
* Install PyTorch: https://pytorch.org/get-started/locally/
* Install Signatory: https://github.com/patrick-kidger/signatory#installation
* Run ```pip install -r requirements.txt``` in the root directory of this project
* Run ```pip install .``` in the root directory of this project

## Usage
Below is a quick summaryof the modules contained in this project and their intended use. All modules should have fairly extensive in-code documentation, which at this point is the recommended way of familiarising yourself with the use of the use of the package. Across the model implementations in the [experiments module](./shar/experiments/models/) and in the [examples folder](./examples/models) you should be able to find an example of almost every component offered by this package in use.

### Experiments
Aside from implementations of various building blocks for shar models from the literature this package provides and experiments module which aims to facilitate basic experimentation with minimal effort. More precisely, it aims to provide a way of running experiments with only having to write down the pure PyTorch model (+3 extra lines of code). The training process can be configured up to a point via comman line arguments and models and a number of datasets can be switched out by commmand line argument to facilitate comparison of performance between models and different datasets. Several models from the literature are provided directly within the module. See the examples [examples readme](./examples) for a more detailed discussion of the options provided by this module and some of its limitations.

### Examples
The projects examples folder contains an example of how to run experiments using the experiments module. Moreover the ST-TR model implemented for the example in examples/models is an example of how to use the general building blocks provided in this model to build a skeleton-based action recognition model. More examples can be found in the experiments module under shar/experiments/models.

### Submodules
All modules contain in-code documentation, you'll find a brief summary of the individual modules below. All network modules will, unless otherwise stated, expect an input of the shape (batch, dimensions, frames, landmarks) and return a tensor of the same shape, i.e. they follow the PyTorch channels first convention. 

#### data
Provides a class SkeletonDataset which provides a Sequence-type interface to the data that can be passed into a PyTorch Dataloader. Can automatically adjust the length of every sample by padding/interpolating in the frame dimension. Returns a fixed number of persons by truncating/padding in the person dimension. The data is returned as a (keypoints, action_label) tuple, with the keypoints tensor of shape (persons, dimensions, frames, landmarks) or optionally (dimensions, frames, landmarks) if only one person is returned (i.e. it follows the PyTorch channels first convention).

#### datatransforms
Provides a person2batch layer to move the person dimension into the batch dimension as is often done to extend models to deal with multiple people. The layers `forward` method moves persons into the batch dimension. To reverse the operation at the end of the models computation the layer provides a method `extract_persons` which extracts persons from the batch dimension and aggregates the results by either taking the mean or max over the persons of each sample.

#### normalisations
Provides ChannelwiseBatchnorm, a version of batch norm which performs batch norm on the individual channels of the landmarks.

#### graphs
Provides a number of modules for graph convolutions.
* The `Graph` class provides an implementation of directed or undirected graphs. It provides the graphs adjacency matrix as return of its `forward` method. This allows it to provide an adjacency matrix that can optionally contain a (additive) data dependent component. Moreover the adjacency matrix can optionally contain a learnable (additive) component or a (multiplicative) edge importance weighting. The adjacency matrix is provided split into several individual matrices according to a given partition strategy of the neighboursets of the nodes. The graph is configured using a GraphLayout object, which cover data dependent structure such as the edges between nodes, and a GraphOptions object, which covers the method dependent options such as what components of the adjacency matrix to use.
(For more information on the components of the adjacency matrix and partitioning strategies see the [ST-GCN paper](https://arxiv.org/pdf/1801.07455.pdf), [AGCN paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf) and [MS-G3D paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Disentangling_and_Unifying_Graph_Convolutions_for_Skeleton-Based_Action_Recognition_CVPR_2020_paper.pdf))
* The `GraphConvolution`, `TemporalGraphConvolution`, `SpatioTemporalGraphConvolution` and `G3DGraphConvolution` classes provide the graph convolutions described and used in the [ST-GCN paper](https://arxiv.org/pdf/1801.07455.pdf), [MS-G3D paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Disentangling_and_Unifying_Graph_Convolutions_for_Skeleton-Based_Action_Recognition_CVPR_2020_paper.pdf) and many follow-up papers.

#### transformers
Provides `SpatialTransformer`, `TemporalTransformer` and `SpatioTemporalTransformer` classes based on the [ST-TR paper](https://arxiv.org/pdf/2008.07404.pdf). The spatial transformer views the landmarks in each frame as a sequence in space, the temporal transformer views each landmark as a sequence in time. To these sequences a standard transfomer model is applied. The spatiotemporal transformer instead views a skeleton sequence as a sequence in time and space, iterating all landmarks in one frame before moving on to the next frame. A standard transformer model is then applied to this model.

#### signatures
Provides the `LogSigRNN` based on the [LogSigRNN paper](https://www.bmvc2021-virtualconference.com/assets/papers/0724.pdf). This class combines a Logsignature transform to summarise the data locally in time to reduce the time dimension with an LSTM. The local summary of the data using the logsignature can help reduce the computational cost of the LSTM for long time series and improves robustness to changes in frame rate and missing data.
Provides the `SegmentSignatures` module to compute signatures or logsignatures of segments in time of each component of spatio-temporal data. More precisely, for a sequence of data of the form `(batch, channels, frames, landmarks/nodes)` the sequence is split into a given number of segments along the frame dimension and then the signature of each segment for each individual landmark/node is computed. This reduces the time dimension considerably using the signature as an efficient summary of short time windows.
Provides the `DyadicSignature` module. This module computes the signatures of all individual segments (in time) at dyadic levels (i.e. signatures of the halves of the sequence, signatures of quarters of the sequence, etc). At each dyadic level the signatures of all segments at that level are combined as a weighted sum based on an attention score. The attention score is computed as a linear function of the segments signatures.

### pathtransformations
Provides a number of path transformations which are common in various areas, namely `AccumulativeTransform`, `TimeIncorporatedTransform`, `MultiDelayedTransform`, `InvisibilityResetTransform`. Each of these transforms augments the path with certain information and in particular can be very valuable with modules from the signature module, as they have been shown to increase the expressiveness of the signature transform.

## Testing
Tests are at this point almost entirely missing, what does exist is based on pytest. The code contains type hints for MyPy.
