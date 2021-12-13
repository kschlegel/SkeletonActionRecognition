# Examples

## Run an experiment
Navigate to the examples subfolder and run
```python experiment.py -ds [DATASET_NAME] -p [PATH TO DATA] --model_name [MODEL]```
The models included at this point are stgcn, agcn, logsigrnn, gcnlogsigrnn and sttr for [ST-GCN](https://arxiv.org/pdf/1801.07455.pdf), [1s-AGCN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf), [LogsigRNN](https://www.bmvc2021-virtualconference.com/assets/papers/0724.pdf) and [1s-ST-TR](https://arxiv.org/pdf/2008.07404.pdf) respectively. The datasets directly supported at this point are NTURGBD, ChaLearn2013, Skeletics152, JHMDB and BerkeleyMHAD (in csv form). The path to the data should be to the base directory to the data of the chosen dataset. Running experiments this way uses the [DatasetLoader](https://github.com/kschlegel/DatasetLoader) package which will attempt to load the data from the given directory, e.g. for the NTURGBD data the dataset loader expects to find in this folder a subfolder `nturgb+d_skeletons` as you obtain when unpacking the data when downloading the dataset. It also expects to find a text file `NTU_RGBD_samples_with_missing_skeletons.txt` (and `NTU_RGBD120_samples_with_missing_skeletons.txt` should you intend to use NTU RGBD 120) containing the ids of sequences with missing skeletons. These are exactly the files that can be obtained [here](https://github.com/shahroudy/NTURGB-D).

The GCN-type models by default will run in a significantly smaller version than presented in their respective papers for easier and quicker testing. To train the full sized model as in the paper use the `--layers` flag when training to increase the number of layers up to the size presented in the paper. For more information on the command line flags, both provided by PyTorch Lightning and the examples run `python experiment.py -h` or `python experiment.py --model_name [MODEL] -h` to include the options provided by this particular model or `python experiment.py -ds [DATASET] -h`.  to include the options provided by the particular dataset.

# Experiments
The purpose of the `Experiments` module is to provide a way running basic experiments at minimal effort. It implements a very standard training procedure for classification task which takes a sequence of skeleton data as input and produces an action class output and provides some basic options for choosing things like the optimiser and metrics used. The options may not be sufficient for supporting a research project from start to finish, but should allow very rapid testing out ideas.

## How to implement and run experiments on your own models
### Implement the model(s)
The `Experiments` class expects to be provided with a (or several) folders containing models you want to be able to train. Every public python file (file with a name not starting with a underscore \"_\" and ending with .py) is assumed to contain a class with the same name up to capitalisation (e.g. LogSigRNN in logsigrnn.py - the filename is assumed to be all lower case following standard python conventions). This model class is essentially a standard PyTorch module:
```
DEFAULT_SOME_OPTION = 42
class MyModel(torch.nn.Module):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("MyModel specific args")
        parser.add_argument("--some_option", type=int, default=DEFAULT_SOME_OPTION, help="Some option for my model")
        return parent_parser
        
    def __init__(self, some_option=DEFAULT_SOME_OPTION, **kwargs):
        super().__init__()
        # init stuff here
        
    def forward(self, x):
        # do stuff with x here
        return x
```
Two things are important to note here:
1. You **have to** add `**kwargs` to the constructor of your model. This is because the model will be _passed all options_ from the command line and some derived values. This allows you to parametrize your model on any options, in particular the model will receive the parameters `keypoint_dim, num_keypoints, num_classes, num_persons` capturing the structure of the data, so that you can parametrize the model on these and allow it to run on any of the supported datasets.
2. You **can** choose to add a static method `add_argparse_args`. If your model does contain this method it will be automatically called by the experiments class and its arguments added to the command line interface, i.e. in the above example you could run something like `python experiment.py -ds [DATASET] -p [PATH_TO_DATA] --model_name mymodel --some_option 5` (As also recommended in the PyTorch Lightning documentation, it is a good idea to provide a reasonable default value for any option you define for your model so that a user can try it out without needing to understand how to choose a value for the given option)

### Run experiments
After defining your model(s) all you need to do is write a very simple driver script
```
from shar.experiments import Experiments
exp = Experiments("./models")
exp.run()
```
You just need to provide the path to your models (or several paths as a list) to the `Experiments` class and call run. The constructor handles everything up to parsing of the command line options, while the call to run will set up the data and model and run the training loop. This means two consecutive calls to run in the same file would result in two independent runs of the experiment with identical settings.

If the driver file is called `experiment.py` you can now run experiments using `python experiment.py [COMMAND LINE ARGS]` as shown above. Experiments are logged with a Tensorboard logger, the logs can be viewed by calling `tensorboard --logdir=lightning_log`. The latest checkpoint for each model is saved in the checkpoints subfolder. Storing checkpoints and logs in separate directories means you can easily discard the checkpoints (large files) after finishing your experiments while keeping the logs (small files) to be able to refer back to them at a later time.

## Options provided by the Experiments class
Below is an overview of most of the command line arguments supported for experiments. You can also get information about these and a few additional arguments by running `python experiment.py -h` to list the supported options or `python experiment.py --model_name [MODEL] -h` to include the options provided by this particular model or `python experiment.py -ds [DATASET] -h`.  to include the options provided by the particular dataset.

### General training options
The experiments module is build on top of PyTorch Lightning and exposes the command line options provided by it to customise the training process. For a detailed description of these options see the [PyTorch Lightning documentation](https://pytorch-lightning.readthedocs.io/en/latest/)

On top of the PyTorch Lightning options the following options are supported:
- `--optimizer` and `--lr` to select the optimizer (adam or sgd) and the learning rate
- `--mAP` to compute the mean average precision on top of accuracy
- `--training_metrics` to also log the metrics for each training steps
- `--confusion_matrix` to log confusion matrices of each validation step to tensorboard
- `--parameter_histograms` to log histograms of parameter and gradient norms at the end of each training step

### Data options
- `-ds/--dataset` allows to select datasets using the [DatasetLoader](https://github.com/kschlegel/DatasetLoader) package and needs to be used in conjunction with the `-p/--data_path` option specifying where the dataset is located. Additionally the DatasetLoader provides `--split` to select a dataset train/val split if there are several available. For some datasets the DatasetLoader provides further options, e.g. taking smaller subsets of NTURGBD
- alternatively the `--data_files` options can be used to load data from a pair of numpy files ([data_files]_training.npy, [data_files]_test.npy) containing the training and test data. This option can also be used in conjunction with the `--dataset` option. This will in general load slightly faster and has the benefit that information about the dataset structure can be obtained from the DatasetLoader (such as number of classes which otw is expensive, or class names for the confusion matrices which otw is not possible). Moreover, if the specified filename does not exist on the first run it will be created and then re-used on any subsequent run. This allows e.g. to create a subset of the NTU RGBD data on the first run of a sequence of experiments and to efficiently re-use the same subset on every subsequent run.
- `--adjust_len` and `--target_len` allow adjusting the length (number of frames) of individual sequences to a fixed given length by a number of methods (looping/padding for up-sampling, interpolation for up&down-sampling)
- `--num_persons` allows to choose the number of persons per sequence to be returned per data sample to allow for datasets with multiple people such as NTU RGBD
- `-b/--batch_size` and `--num_workers` to select batch size and number of workers for the data loaders
- For dataset specific arguments run `python experiment.py -ds [DATASET] -h`

### Experiment options
- `--model_name` lets you choose which model to train. These are all models implemented as described above in the folder(s) you passed to the constructor, plus some build-in models always available. At this point the build-in models are [stgcn](https://arxiv.org/pdf/1801.07455.pdf), [agcn](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf), [logsigrnn and gcnlogsigrnn](https://www.bmvc2021-virtualconference.com/assets/papers/0724.pdf)
- For model specific arguments run `python experiment.py --model_name [MODEL_NAME] -h`
- `-x/--experiment_name` lets you specify an additional name for the experiment. In this case the logs are stored in the subfolder _./lightning_logs_[EXPERIMENT_NAME]_ and checkpoints are saved in the subfolder _./checkpoints_[EXPERIMENT_NAME]_. This allows grouping experiments by different parameter combinations to be explored. As mentioned before, the separate directories allow discarding large checkpoint files while keeping the small log files for reference.
- `--additional` This parameter has no direct effect on the training run but is added to the log files as a hyperparameter, so that it can be used to include additional information into the logs that isn't captured well by the hyperparameters logged
