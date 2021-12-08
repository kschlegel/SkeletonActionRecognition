from typing import Sequence, Optional, Tuple, Union
from warnings import warn

import numpy as np
import cv2
from torch.utils.data import Dataset

from shar._utils.argparser import WithDefaultsWrapper


class SkeletonDataset(Dataset):
    """
    Torch dataset class that can be passed into a dataloader.

    Expects a Sequence-type interface (providing __len__ and __getitem__) to
    the data, which returns the data as (keypoints, action_class) tuples, with
    the keypoints in a (persons,frames,landmarks,dimensios) format. Optionally
    can also take a str to a numpy file containing the data in the same format,
    i.e. an array of shape (batch, [keypoints,label],...) where [batch,0]
    contains the keypoint array and [batch,1] the action id.
    Returns samples as tuples (keypoints, action_class) with the keypoints of
    shape either
    (persons, dimensions, frames, landmarks) or (dimensions, frames, landmarks)
    with the latter only occuring when both num_persons == 1 and
    keep_person_dim == False.
    The person dimension is automatically equalised to the selected number by
    truncation/padding with zeros as appropriate.
    Can optionally re-scale the time dimension to a fixed length, using either
    linear interpolation, looping of the sequence or padding with zeros or the
    last frame. Warning: Only interpolation will currently also deal with
    sequences shorter than the target length.
    """
    @staticmethod
    def add_argparse_args(parser,
                          default_adjust_len: str = "interpolate",
                          default_target_len: Optional[str] = None,
                          default_num_persons: int = 2):
        if isinstance(parser, WithDefaultsWrapper):
            local_parser = parser
        else:
            local_parser = WithDefaultsWrapper(parser)
        local_parser.add_argument(
            '--adjust_len',
            type=str,
            choices=["interpolate", "loop", "pad_zero", "pad_last"],
            default=default_adjust_len,
            help="Adjust the length of individual sequences to a common length"
            " by interpolation, looping the sequence or padding with either "
            "zeros or the last frame.")
        local_parser.add_argument(
            '-l',
            '--target_len',
            type=int,
            default=default_target_len,
            help="Number of frames to scale action sequences to")
        local_parser.add_argument(
            '--num_persons',
            type=int,
            default=default_num_persons,
            help="Number of people to return (extra persons are discarded, "
            "missing persons zero padded)")
        local_parser.add_argument(
            '--keep_person_dim',
            action="store_true",
            help="Only relevevant if num_persons == 1. In that case if set "
            "the keypoint data is returned as a 4D array with a person-"
            "dimension of size 1. If not set the keypoint data is returned as "
            "a 3D array without a person dimension.")

        return parser

    def __init__(self,
                 data: Union[Sequence, str],
                 adjust_len: Optional[str] = None,
                 target_len: Optional[int] = None,
                 num_persons: int = 2,
                 keep_person_dim: bool = False,
                 **kwargs) -> None:
        """
        Parameters
        ----------
        data : Sequence or str
            A Sequence type object (providing __len__ and __getitem__)
            providing access to the data. Or alternatively a filename of a
            numpy file containing the data in appropriate shape (see class doc
            for a description)
        adjust_len : str, optional (default is None)
            One of ('interpolate', 'loop', 'pad_zero', 'pad_last')
            Optionally adjust the length of each sequence to a fixed number of
            frames by either linear interpoaltion, looping the sequence,
            padding at the end with zeros or padding at the end with the last
            frame. If set target_len must be specified.
        target_len : int, optional (default is None)
            Length to adjust every sample to. Only used if adjust_len is not
            None. Must be specified if adjust_len is not None.
        num_persons : int, optional (default is 2)
            The returned keypoint array is for exactly this number of persons.
            Extra entries in the data are discarded, if fewer skeletons exist
            zeros are aded as padding.
        keep_person_dim : bool, optional (default is False)
            Only relevevant if num_persons == 1. In that case if True the
            keypoint data is returned as a 4D array with a person-dimension of
            size 1. If False the keypoint data is returned as a 3D array
            without a person dimension.
        """
        if isinstance(data, str):
            self._data = np.load(data, allow_pickle=True)
        else:
            self._data = data

        if adjust_len is not None and target_len is None:
            raise ValueError("Target length must be specified when selecting "
                             "to adjust length of samples")
        self._adjust_len = adjust_len
        self._target_len = target_len
        self._num_persons = num_persons
        self._keep_person_dim = keep_person_dim

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        keypoints, action = self._data[index]
        if len(keypoints.shape) == 3:
            # if data has no person dimension temporarily add one
            keypoints = np.expand_dims(keypoints, 0)
        # keypoints = (person, frame, landmark, coordinates)

        # optionally adjust length
        if self._adjust_len is None:
            pass
        elif self._adjust_len == "interpolate":
            # Linearly interpolate the frame dimension
            shape = (keypoints.shape[2], self._target_len)
            rescaled = []
            for i in range(keypoints.shape[0]):
                rescaled.append(
                    cv2.resize(keypoints[i],
                               shape,
                               interpolation=cv2.INTER_LINEAR))
            keypoints = np.stack(rescaled, axis=0)
        elif self._adjust_len == "loop":
            # Loop the frame dimension, repeating the sequence as many times as
            # necessary
            padding_size = self._target_len - keypoints.shape[1]
            full_loops = padding_size // keypoints.shape[1]
            if full_loops > 0:
                padding_size -= full_loops * keypoints.shape[1]
                padding = np.repeat(keypoints, full_loops, axis=1)
                keypoints = np.concatenate((keypoints, padding), axis=1)
            keypoints = np.concatenate(
                (keypoints, keypoints[:, :padding_size]), axis=1)
        elif self._adjust_len.startswith("pad"):
            # Pad the sequence at the end with zeros or the last frame
            padding_size = self._target_len - keypoints.shape[1]
            if self._adjust_len.endswith("zero"):
                padding = np.zeros(
                    (keypoints.shape[0], padding_size) + keypoints.shape[2:],
                    dtype=keypoints.dtype)
            elif self._adjust_len.endswith("last"):
                padding = np.expand_dims(keypoints[:, -1], 1)
                padding = np.repeat(padding, padding_size, axis=1)
            keypoints = np.concatenate((keypoints, padding), axis=1)

        # reorder for PyTorch channels first convention
        #    -> new order: (person, coordinates, frame, landmark)
        keypoints = keypoints.transpose((0, 3, 1, 2))

        # adjust person dimension if need be
        if not self._keep_person_dim and self._num_persons == 1:
            keypoints = keypoints[0]
        elif keypoints.shape[0] < self._num_persons:
            keypoints = np.concatenate(
                (keypoints,
                 np.zeros((self._num_persons - keypoints.shape[0], ) +
                          keypoints.shape[1:],
                          dtype=keypoints.dtype)),
                axis=0)
        elif keypoints.shape[0] > self._num_persons:
            keypoints = keypoints[:self._num_persons]

        return np.ascontiguousarray(keypoints, dtype=np.float32), action

    def get_num_keypoints(self):
        if len(self._data) == 0:
            raise ValueError("No data to determine number of keypoints.")
        else:
            keypoints, __ = self._data[0]
            # keypoints = (frame, landmark, coordinates) or
            #             (person, frame, landmark, coordinates)
        return keypoints.shape[-2]

    def get_num_actions(self):
        if len(self._data) == 0:
            raise ValueError("No data to determine number of keypoints.")
        else:
            if isinstance(self._data, np.ndarray):
                max_action = np.amax(self._data[:, 1])
            else:
                warn(
                    "get_num_actions from sequence object is very inefficient!"
                )
                max_action = 0
                for __, action in self._data:
                    max_action = max(max_action, action)
            return max_action + 1

    def get_keypoint_dim(self):
        if len(self._data) == 0:
            raise ValueError("No data to determine number of keypoints.")
        else:
            keypoints, __ = self._data[0]
            # keypoints = (frame, landmark, coordinates) or
            #             (person, frame, landmark, coordinates)
        return keypoints.shape[-1]
