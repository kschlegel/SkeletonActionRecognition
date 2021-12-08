import torch

from shar.datatransforms import Person2Batch
from shar.normalisations import ChannelwiseBatchNorm
from shar.pathtransformations import (AccumulativeTransform, EmbeddingLayer,
                                      TimeIncorporatedTransform)
from shar.signatures import LogSigRNN as LogSigRNNLayer

DEFAULT_NUM_SEGMENTS = 50
DEFAULT_LSTM_CHANNELS = 96


class LogSigRNN(torch.nn.Module):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("LogSigRNN specific")
        parser.add_argument('--num_segments',
                            type=int,
                            default=DEFAULT_NUM_SEGMENTS,
                            help="Number of segments")
        parser.add_argument('--lstm_channels',
                            type=int,
                            default=DEFAULT_LSTM_CHANNELS,
                            help="LSTM output channels")
        parser.add_argument(
            '--embedding_layer',
            type=int,
            help="Include Embedding layer, value determines the dimension of "
            "the output of the embedding layer.")
        return parent_parser

    def __init__(self,
                 keypoint_dim,
                 num_keypoints,
                 num_classes,
                 num_persons,
                 num_segments=DEFAULT_NUM_SEGMENTS,
                 lstm_channels=DEFAULT_LSTM_CHANNELS,
                 embedding_layer=None,
                 **kwargs):
        super().__init__()

        self.data_batch_norm = ChannelwiseBatchNorm(in_channels=keypoint_dim,
                                                    landmarks=num_keypoints)
        self.person2batch = Person2Batch(person_dimension=1,
                                         num_persons=num_persons)

        self.embedding_layer = embedding_layer
        if embedding_layer is not None:
            self.embedding_layer = EmbeddingLayer(in_channels=keypoint_dim,
                                                  out_channels=embedding_layer,
                                                  landmarks=num_keypoints)
            self.accumulative = AccumulativeTransform()
            self.time_incorporated = TimeIncorporatedTransform()
            logsigrnn_inchannels = 31
        else:
            logsigrnn_inchannels = keypoint_dim * num_keypoints

        self.logsigrnn = LogSigRNNLayer(in_channels=logsigrnn_inchannels,
                                        logsignature_lvl=2,
                                        num_segments=num_segments,
                                        out_channels=lstm_channels,
                                        include_startpoint=True)

        self.fully_connected = torch.nn.Linear(lstm_channels * num_segments,
                                               num_classes)

    def forward(self, x):
        # Move persons into the batch dimension
        x = self.person2batch(x)
        # Normalise data
        x = self.data_batch_norm(x)

        if self.embedding_layer:
            x = self.embedding_layer(x)
            x = self.accumulative(x)
            x = self.time_incorporated(x)
        else:
            # view the skeleton as one high dimensional point
            # (batch, channels, frames, landmarks)
            x = x.permute(0, 1, 3, 2)
            x = torch.reshape(x, (x.shape[0], -1, x.shape[3], 1))
            # (batch, channels*landmarks, frames, 1)

        x = self.logsigrnn(x)

        x = self.person2batch.extract_persons(x)
        x = torch.flatten(x, start_dim=1)

        # Predict
        x = self.fully_connected(x)

        return x
