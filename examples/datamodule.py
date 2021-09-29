import sys
sys.path.insert(0, '')
sys.path.append('..')

from torch.utils.data import DataLoader
import pytorch_lightning as pl
from datasetloader import NTURGBD

from shar.data import SkeletonDataset

DEFAULT_BATCH_SIZE = 64
DEFAULT_ADJUST_LEN = "interpolate"
DEFAULT_TARGET_LEN = 300


class SkeletonDataModule(pl.LightningDataModule):
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data specific arguments")
        NTURGBD.add_args(parser)

        parser.add_argument('-b',
                            '--batch_size',
                            type=int,
                            default=DEFAULT_BATCH_SIZE,
                            help="Batch size to use (default is {})".format(
                                DEFAULT_BATCH_SIZE))
        parser.add_argument(
            '--adjust_len',
            type=str,
            choices=["interpolate", "loop", "pad_zero", "pad_last"],
            default=DEFAULT_ADJUST_LEN,
            help="Adjust the length of individual sequences to a common length"
            " by interpolation, looping the sequence or padding with either "
            "zeros or the last frame (default is {})".format(
                DEFAULT_ADJUST_LEN))
        parser.add_argument('-l',
                            '--target_len',
                            type=int,
                            default=DEFAULT_TARGET_LEN,
                            help="Number of frames to scale action sequences "
                            "to (default is {})".format(DEFAULT_TARGET_LEN))

        parser.add_argument(
            '--num_workers',
            type=int,
            default=4,
            help="Number of workers to use for dataloaders (default is 4).")

        return parent_parser

    def __init__(self, **kwargs):
        super().__init__()
        self._data = NTURGBD(kwargs["path"])
        self._data.set_cols("keypoints3D", "action")
        self._data.set_split(kwargs["split"])

        self._batch_size = kwargs["batch_size"]
        self._adjust_len = kwargs["adjust_len"]
        self._target_len = kwargs["target_len"]
        self._num_workers = kwargs["num_workers"]

    def setup(self, stage=None):
        self._trainingdata = SkeletonDataset(data=self._data.trainingset,
                                             adjust_len=self._adjust_len,
                                             target_len=self._target_len)
        self._testdata = SkeletonDataset(data=self._data.testset,
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
