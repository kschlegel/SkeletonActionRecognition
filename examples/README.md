# Examples

## Requirements
The examples have additional dependencies on [PyTorch Lightning](https://www.pytorchlightning.ai/) and my [DatasetLoader package](https://github.com/kschlegel/DatasetLoader). These can be installed by running
```pip install -r examples/requirements.txt```

## Usage
To run the examples first install the additional requirements. Then navigate to the examples subfolder and run
```python train.py --model_name [MODEL] -p [PATH TO DATA]```
The example models included at this point are stgcn, agcn and sttr for [ST-GCN](https://arxiv.org/pdf/1801.07455.pdf), [1s-AGCN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf) and [1s-ST-TR](https://arxiv.org/pdf/2008.07404.pdf) respectively. The path to the data should be to the base directory to the NTU RGBD data. The dataset loader expects to find in this folder a subfolder `nturgb+d_skeletons` as you obtain when unpacking the data when downloading the dataset. It also expects to find a text file `NTU_RGBD_samples_with_missing_skeletons.txt` (and `NTU_RGBD120_samples_with_missing_skeletons.txt` should you intend to use NTU RGBD 120) containing the ids of sequences with missing skeletons. These are exactly the files that can be obtained [here](https://github.com/shahroudy/NTURGB-D).

All models by default will run in a significantly smaller version than presented in their respective papers for easier and quicker testing. To train the full sized model as in the paper use the `--full_model` flag when training. For mor information on the command line flags, both provided by PyTorch Lightning and the examples run `python train.py -h` or `python train.py --model_name [MODEL] -h` to include the options provided by this particular model. 
