import torch
from shar.datatransforms import Person2Batch
from shar.normalisations import ChannelwiseBatchNorm
from shar.graphs import SpatioTemporalGraphConvolution
from shar.graphs.graphlayouts import get_layout_by_datasetname
from shar.graphs.graphoptions import ST_GCN_Options

DEFAULT_NUM_LAYERS = 3
DEFAULT_PARTITION_STRATEGY = "uniform"


class STGCN(torch.nn.Module):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("ST-GCN specific arguments")
        parser.add_argument('--edge_importance_weighting',
                            action='store_true',
                            help="Add a learnable multiplicative edge "
                            "importance weighting to the adjacency matrix.")
        parser.add_argument(
            '--partition_strategy',
            type=str,
            choices=["uniform", "distance", "spatial"],
            default=DEFAULT_PARTITION_STRATEGY,
            help="Partition strategy used to partition node neighbour sets.")
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
                 partition_strategy=DEFAULT_PARTITION_STRATEGY,
                 edge_importance_weighting=False,
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
            ST_GCN_Options(partition_strategy=partition_strategy,
                           edge_importance_weighting=edge_importance_weighting)
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

        self.fully_connected = torch.nn.Conv2d(channels[layers],
                                               num_classes,
                                               kernel_size=1)

    def forward(self, x):
        # Move persons into the batch dimension
        x = self.person2batch(x)

        # Normalise data
        x = self.data_batch_norm(x)

        # Forwad
        for gcn in self.gcn_networks:
            x = gcn(x)

        # Global pooling
        x = torch.nn.functional.avg_pool2d(x, x.size()[2:])

        # Aggregate results for people of each batch element
        x = self.person2batch.extract_persons(x)

        # Predict
        x = self.fully_connected(x)
        x = x.view(x.size(0), -1)
        return x
