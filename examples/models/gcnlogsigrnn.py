import torch

from shar.datatransforms import Person2Batch
from shar.normalisations import ChannelwiseBatchNorm
from shar.signatures import LogSigRNN
from shar.graphs import GraphConvolution
from shar.graphs.graphlayouts import KinectV2
from shar.graphs.graphoptions import MS_GCN_Options


class GCNLogSigRNN(torch.nn.Module):
    @staticmethod
    def add_stgcn_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GCNLogSigRNN specific")
        parser.add_argument('--layers',
                            type=int,
                            default=1,
                            choices=[1, 2],
                            help="Number of GCN+LogSigRNN blocks.")

    def __init__(self, num_classes, num_gcn_scales=13, layers=1, **kwargs):
        super().__init__()

        self.data_batch_norm = ChannelwiseBatchNorm(in_channels=3,
                                                    landmarks=25)
        self.person2batch = Person2Batch(person_dimension=1, num_persons=2)

        graph = {
            "graph_layout": KinectV2,
            "graph_options":
            MS_GCN_Options(max_neighbour_distance=num_gcn_scales)
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
                    logsignature_lvl=2,
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
