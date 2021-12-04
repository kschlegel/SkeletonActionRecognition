# SkeletonActionRecognition
A collection of skeleton-based human action recognition methods using PyTorch

## Contents
1. [Installation](#installation)
2. [Usage](#usage)
   1. [Examples](#examples)
   2. [Submodules](#submodules)
3. [Testing](#testing)

## Installation
* Run ```pip install -r requirements.txt``` in the root directory of this project
* Install PyTorch: https://pytorch.org/get-started/locally/
* Install Signatory: https://github.com/patrick-kidger/signatory#installation
* Run ```pip install .``` in the root directory of this project

## Usage
### Examples
The project contains an example how to train a number of models on the NTU RGBD 60 dataset. More details on how to run the examples can be found in the [examples readme](./examples). The implementation of the individual models can be found in [examples/models](./examples/models) if you prefer to have a look at the code.

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
* The `Graph` class provides an implementation of directed or undirected graphs. It provides the graphs adjacency matrix as return of its `forward` method. This allows it to provide an adjacency matrix that can optionally contain a (additive) data dependent component. Moreover the adjacency matrix can optionally contain a learnable (additive) component or a (multiplicative) edge importance weighting. The adjacency matrix is provided split into several individual matrices according to a given partition strategy of the neighboursets of the nodes.
(For more information on the components of the adjacency matrix and partitioning strategies see the [ST-GCN paper](https://arxiv.org/pdf/1801.07455.pdf) and the [AGCN paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf))
* The `GraphConvolution`, `TemporalGraphConvolution` and `SpatioTemporalGraphConvolution` classes provide the graph convolutions described and used in the [ST-GCN paper](https://arxiv.org/pdf/1801.07455.pdf) and many follow-up papers.

#### transformers
Provides `SpatialTransformer`, `TemporalTransformer` and `SpatioTemporalTransformer` classes based on the [ST-TR paper](https://arxiv.org/pdf/2008.07404.pdf). The spatial transformer views the landmarks in each frame as a sequence in space, the temporal transformer views each landmark as a sequence in time. To these sequences a standard transfomer model is applied. The spatiotemporal transformer instead views a skeleton sequence as a sequence in time and space, iterating all landmarks in one frame before moving on to the next frame. A standard transformer model is then applied to this model.

#### signatures
Provides the `DyadicSignature` module. This module computes the signatures of all individual segments (in time) at dyadic levels (i.e. signatures of the halves of the sequence, signatures of quarters of the sequence, etc). At each dyadic level the signatures of all segments at that level are combined as a weighted sum based on an attention score. The attention score is computed as a linear function of the segments signatures.

## Testing
Tests are at this point almost entirely missing, what does exist is based on pytest. The code contains type hints for MyPy.
