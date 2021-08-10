import numpy as np
import cv2
from torch.utils.data import Dataset
from datasetloader import NTURGBD


class SkeletonDataset(Dataset):
    """
    Torch dataset class that can be passed into a dataloader.

    Provides an interface to the data that can be passed into the data loader.
    Returns samples as tuples (keypoints, action) with the keypoints of shape
    (persons, dimensions, frames, landmarks)
    The person dimension is automatically equalised to a selected number by
    truncation/padding with zeros as appropriate.
    Can optionally re-scale the time dimension to a fixed length, using either
    linear interpolation, looping of the sequence or padding with zeros or the
    last frame.
    """
    def __init__(self,
                 data_path,
                 split,
                 subset,
                 adjust_len=None,
                 target_len=None,
                 num_persons=2):
        """
        Parameters
        ----------
        data_path : str
            Path to the dataset
        split : str
            Dataset split to use - One of ("cross-view", "cross-subject")
        subset : str
            Data subset - one of ("train", "test")
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
        """
        self._data = NTURGBD(data_path)
        self._data.set_cols("keypoints3D", "action")
        self._samples = self._data.get_split(split, subset)

        if adjust_len is not None and target_len is None:
            raise ValueError("Target length must be specified when selecting "
                             "to adjust length of samples")
        self._adjust_len = adjust_len
        self._target_len = target_len
        self._num_persons = num_persons

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        sample = self._data[self._samples[index]]
        # sample["keypoints3D"] = (person, frame, landmark, coordinates)

        # optionally adjust length
        if self._adjust_len is None:
            keypoints = sample["keypoints3D"]
        elif self._adjust_len == "interpolate":
            # Linearly interpolate the frame dimension
            shape = (sample["keypoints3D"].shape[2], self._target_len)
            rescaled = []
            for i in range(sample["keypoints3D"].shape[0]):
                rescaled.append(
                    cv2.resize(sample["keypoints3D"][i],
                               shape,
                               interpolation=cv2.INTER_LINEAR))
            keypoints = np.stack(rescaled, axis=0)
        elif self._adjust_len == "loop":
            # Loop the frame dimension, repeating the sequence as many times as
            # necessary
            keypoints = sample["keypoints3D"]
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
            padding_size = self._target_len - sample["keypoints3D"].shape[1]
            if self._adjust_len.endswith("zero"):
                padding = np.zeros(
                    (sample["keypoints3D"].shape[0], padding_size) +
                    sample["keypoints3D"].shape[2:],
                    dtype=sample["keypoints3D"].dtype)
            elif self._adjust_len.endswith("last"):
                padding = np.expand_dims(sample["keypoints3D"][:, -1], 1)
                padding = np.repeat(padding, padding_size, axis=1)
            keypoints = np.concatenate((sample["keypoints3D"], padding),
                                       axis=1)

        # reorder for PyTorch channels first convention
        #    -> new order: (person, coordinates, frame, landmark)
        keypoints = keypoints.transpose((0, 3, 1, 2))

        # adjust person dimension if need be
        if keypoints.shape[0] < self._num_persons:
            keypoints = np.concatenate(
                (keypoints,
                 np.zeros((self._num_persons - keypoints.shape[0], ) +
                          keypoints.shape[1:],
                          dtype=keypoints.dtype)),
                axis=0)
        elif keypoints.shape[0] > self._num_persons:
            keypoints = keypoints[:self._num_persons]

        return np.ascontiguousarray(keypoints,
                                    dtype=np.float32), sample["action"]
