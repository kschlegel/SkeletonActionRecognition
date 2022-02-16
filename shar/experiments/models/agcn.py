import torch
from shar.datatransforms import Person2Batch
from shar.normalisations import ChannelwiseBatchNorm
from shar.graphs import SpatioTemporalGraphConvolution
from shar.graphs.graphlayouts import get_layout_by_datasetname
from shar.graphs.graphoptions import AGCN_Options

DEFAULT_NUM_LAYERS = 3


class AGCN(torch.nn.Module):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("AGCN specific arguments")
        parser.add_argument('--no_learnable_adjacency',
                            action='store_true',
                            help="Add a learnable component to the adjacency "
                            "matrix.")
        parser.add_argument('--no_data_dependent_adjacency',
                            action='store_true',
                            help="Add a data dependent component to the "
                            "adjacency matrix.")
        parser.add_argument(
            '--layers',
            type=int,
            choices=list(range(1, 11)),
            default=DEFAULT_NUM_LAYERS,
            help="Number of GCN layers to use. This allows to train more "
            "lightweight models than the original paper (layers=10 is as "
            "presented in the paper.)")
        parser.add_argument(
            '--graph_layout',
            type=str,
            help="Select a graph layout to use. When using the DatasetLoader "
            "package to load the data this is inferred automaticaly and this "
            "option is ignored. Options are the same as the dataset command "
            "line options.")
        return parent_parser

    def __init__(self,
                 keypoint_dim,
                 num_keypoints,
                 num_classes,
                 num_persons,
                 no_learnable_adjacency=False,
                 no_data_dependent_adjacency=False,
                 layers=DEFAULT_NUM_LAYERS,
                 graph_layout=None,
                 **kwargs):
        super().__init__()

        if kwargs["dataset"] is not None:
            graph_layout = get_layout_by_datasetname(kwargs["dataset"])
        else:
            graph_layout = get_layout_by_datasetname(graph_layout)

        graph = {
            "graph_layout":
            graph_layout,
            "graph_options":
            AGCN_Options(
                learnable_adjacency=not no_learnable_adjacency,
                data_dependent_adjacency=not no_data_dependent_adjacency)
        }

        self.data_batch_norm = ChannelwiseBatchNorm(in_channels=keypoint_dim,
                                                    landmarks=num_keypoints)
        self.person2batch = Person2Batch(person_dimension=1,
                                         num_persons=num_persons)

        temporal_kernel_size = 9
        # Define output_channels for all layers, first item is the input dim of
        # the very first layer. This in particular makes channels[layers] the
        # output dim of the last gcn layer
        channels = [keypoint_dim, 64, 64, 64, 64, 128, 128, 128, 256, 256, 256]
        module_list = []
        for i in range(layers):
            module_list += [
                SpatioTemporalGraphConvolution(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_stride=1,
                    residual=False,
                    **graph)
            ]
        self.gcn_networks = torch.nn.ModuleList(module_list)

        self.fully_connected = torch.nn.Linear(channels[layers], num_classes)

    def forward(self, x):
        # Move persons into the batch dimension
        x = self.person2batch(x)

        # Normalise data
        x = self.data_batch_norm(x)

        # Forwad
        for gcn in self.gcn_networks:
            x = gcn(x)

        # average results accross nodes and remaining frames
        x = x.view(x.shape[:2] + (-1, ))
        # (batch*person, channels, frame*node)
        x = x.mean(2)

        # Aggregate results for people of each batch element
        x = self.person2batch.extract_persons(x)
        # Predict
        x = self.fully_connected(x)
        return x
