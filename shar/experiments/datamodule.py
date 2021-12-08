import os
import sys
import importlib

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from tqdm import trange

from shar.data import SkeletonDataset
from shar._utils.argparser import WithDefaultsWrapper

# these refer to classes in the DatasetLoader package
SUPPORTED_DATASETS = [
    "NTURGBD", "ChaLearn2013", "Skeletics152", "JHMDB", "BerkeleyMHAD"
]


class SkeletonDataModule(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_parser,
                               default_batch_size=32,
                               default_target_len=100):
        if isinstance(parent_parser, WithDefaultsWrapper):
            local_parser = parent_parser
        else:
            local_parser = WithDefaultsWrapper(parent_parser)
        parser = local_parser.add_argument_group("Data specific arguments")
        parser.add_argument(
            '-ds',
            '--dataset',
            type=str,
            choices=SUPPORTED_DATASETS,
            help="Dataset to train on. Use -h with -ds specified to include "
            "arguments specific to the given dataset. Can be used in "
            "combination with data_files argument to load data from the "
            "specified file but use the DatasetLoader to provide information "
            "about the data such as class labels.")
        parser.add_argument(
            '--data_files',
            type=str,
            help="A path to a pair numpy files ([data_files]_training.npy, "
            "[data_files]_test.npy) containing the training and test data.")
        parser.add_argument('-b',
                            '--batch_size',
                            type=int,
                            default=default_batch_size,
                            help="Batch size to use")
        parser.add_argument('--num_workers',
                            type=int,
                            default=4,
                            help="Number of workers to use for dataloaders.")

        # Adds args for adjusting sequence length and number of persons
        SkeletonDataset.add_argparse_args(
            parser, default_target_len=default_target_len)
        parser.add_argument(
            '--keypoint_dim',
            type=int,
            choices=[2, 3],
            default=3,
            help="Whether to use 2D or 3D keypoint data. Only used if the "
            "chosen dataset provides both kinds of keypoints.")

        for i, arg in enumerate(sys.argv):
            if arg == "-ds" or arg == "--dataset":
                dataset = sys.argv[i + 1]
                break
        else:
            dataset = None
        if dataset is not None:
            dataset_module = SkeletonDataModule._get_datasetloader_module(
                dataset)
            dataset_module.add_argparse_args(parent_parser)

        return parent_parser

    def __init__(self, **kwargs):
        super().__init__()
        if kwargs["data_files"] is None and kwargs["dataset"] is None:
            raise Exception(
                "At least on of --dataset and --data_files must be specified")

        if kwargs["data_files"]:
            self._trainingset = kwargs["data_files"] + "_training.npy"
            self._testset = kwargs["data_files"] + "_test.npy"
        if kwargs["dataset"] is not None:
            dataset_module = self._get_datasetloader_module(kwargs["dataset"])
            self._data_loader = dataset_module(**kwargs)
            if kwargs["data_files"] is None:
                if (self._data_loader.has_col("keypoints3D")
                        and self._data_loader.has_col("keypoints2D")):
                    self.keypoint_dim = kwargs["keypoint_dim"]
                elif self._data_loader.has_col("keypoints3D"):
                    self.keypoint_dim = 3
                elif self._data_loader.has_col("keypoints2D"):
                    self.keypoint_dim = 2
                else:
                    raise Exception("Dataset does not have valid keypoints")
                self._data_loader.set_cols(
                    "keypoints{}D".format(self.keypoint_dim), "action")

                self._trainingset = self._data_loader.trainingset
                self._testset = self._data_loader.testset
            else:
                self.create_datafiles(kwargs["data_files"])
                # If we load the data from file the keypoint dim depend on what
                # was selected at the time the file was created, not the value
                # now (create_datafiles only creates files that didn't exist
                # before)
                self.keypoint_dim = None
            self.num_keypoints = len(self._data_loader.landmarks)
            self.num_actions = len(self._data_loader.actions)
            self.class_labels = self._data_loader.actions
        else:
            # If loading from data_files without a DatasetLoader specified we
            # need to figure these out when the data is loaded in setup()
            # (class_labels will simply default to integers in this case)
            self.keypoint_dim = None
            self.num_keypoints = None
            self.num_actions = None
            self.class_labels = None

        self._batch_size = kwargs["batch_size"]
        self._adjust_len = kwargs["adjust_len"]
        self._target_len = kwargs["target_len"]
        self._num_workers = kwargs["num_workers"]

    def setup(self, stage=None):
        self._trainingdata = SkeletonDataset(data=self._trainingset,
                                             adjust_len=self._adjust_len,
                                             target_len=self._target_len)
        if self.keypoint_dim is None:
            self.keypoint_dim = self._trainingdata.get_keypoint_dim()
        if self.num_keypoints is None:
            self.num_keypoints = self._trainingdata.get_num_keypoints()
        if self.num_actions is None:
            self.num_actions = self._trainingdata.get_num_actions()
        self._testdata = SkeletonDataset(data=self._testset,
                                         adjust_len=self._adjust_len,
                                         target_len=self._target_len)

    def train_dataloader(self):
        train_loader = DataLoader(self._trainingdata,
                                  batch_size=self._batch_size,
                                  num_workers=self._num_workers,
                                  shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self._testdata,
                                batch_size=self._batch_size,
                                num_workers=self._num_workers,
                                shuffle=False)
        return val_loader

    @staticmethod
    def _get_datasetloader_module(module_name):
        if module_name is not None:
            module_path = "datasetloader." + module_name.lower()
            if importlib.util.find_spec(module_path) is not None:
                module = importlib.import_module(module_path)
                return getattr(module, module_name)
        raise Exception("Invalid dataset!")

    def create_datafiles(self, data_files):
        for datapart in ("training", "test"):
            filename = data_files + "_" + datapart + ".npy"
            if not os.path.exists(filename):
                samples = []
                subset = eval("self._data_loader." + datapart + "set")
                for i in trange(len(subset),
                                desc=f"Creating {datapart}set file"):
                    sample = subset[i]
                    samples.append(
                        np.array([sample[0], sample[1]], dtype=object))
                data = np.array(samples, dtype=object)
                np.save(filename, data)
