import torch

from shar.datatransforms import Person2Batch
from shar.normalisations import ChannelwiseBatchNorm
from shar.signatures import LogSigRNN
from shar.graphs import GraphConvolution
from shar.graphs.graphlayouts import KinectV2
from shar.graphs.graphoptions import MS_GCN_Options

DEFAULT_NUM_LAYERS = 1
DEFAULT_LOGSIG_LVL = 2
DEFAULT_MAX_NEIGHBOUR_DISTANCE = 13


class GCNLogSigRNN(torch.nn.Module):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group(
            "GCNLogSigRNN specific arguments")
        parser.add_argument('--logsig_lvl',
                            type=int,
                            default=DEFAULT_LOGSIG_LVL,
                            help="Level of truncation of the log-signature.")
        parser.add_argument(
            '--max_neighbour_distance',
            type=int,
            default=DEFAULT_MAX_NEIGHBOUR_DISTANCE,
            help="Maximal distance between nodes of the skeleton graph for "
            "the nodes to be considered neighbours.")
        parser.add_argument('--layers',
                            type=int,
                            default=DEFAULT_NUM_LAYERS,
                            choices=[1, 2],
                            help="Number of GCN+LogSigRNN blocks.")
        return parent_parser

    def __init__(self,
                 keypoint_dim,
                 num_keypoints,
                 num_classes,
                 num_persons,
                 logsig_lvl=DEFAULT_LOGSIG_LVL,
                 max_neighbour_distance=DEFAULT_MAX_NEIGHBOUR_DISTANCE,
                 layers=DEFAULT_NUM_LAYERS,
                 **kwargs):
        super().__init__()

        self.data_batch_norm = ChannelwiseBatchNorm(in_channels=keypoint_dim,
                                                    landmarks=num_keypoints)
        self.person2batch = Person2Batch(person_dimension=1,
                                         num_persons=num_persons)

        graph = {
            "graph_layout":
            KinectV2,
            "graph_options":
            MS_GCN_Options(max_neighbour_distance=max_neighbour_distance)
        }

        channels = [3, 96, 192]
        num_segments = [50, 30]

        module_list = []
        for i in range(layers):
            # TODO: Check options, e.g. residual
            module_list += [
                GraphConvolution(in_channels=channels[i],
                                 out_channels=channels[i + 1],
                                 **graph)
            ]

            module_list += [
                LogSigRNN(
                    in_channels=channels[i + 1],
                    logsignature_lvl=logsig_lvl,
                    num_segments=num_segments[i],
                    out_channels=channels[i + 1],
                )
            ]
        self.gcnlogsigrnn_blocks = torch.nn.ModuleList(module_list)

        self.fc = torch.nn.Linear(channels[layers], num_classes)

    def forward(self, x):
        x = self.person2batch(x)

        x = self.data_batch_norm(x)

        for block in self.gcnlogsigrnn_blocks:
            x = block(x)

        # Global Average Pooling (Spatial+Temporal)
        x = torch.flatten(x, start_dim=2)
        # # (batch*person, channels, frame*node)
        x = x.mean(2)

        x = self.person2batch.extract_persons(x)

        x = self.fc(x)
        return x
