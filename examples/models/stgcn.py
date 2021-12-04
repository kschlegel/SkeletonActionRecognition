import torch
from shar.datatransforms import Person2Batch
from shar.normalisations import ChannelwiseBatchNorm
from shar.graphs import SpatioTemporalGraphConvolution
from shar.graphs.graphlayouts import KinectV2
from shar.graphs.graphoptions import ST_GCN_Options


class STGCN(torch.nn.Module):
    @staticmethod
    def add_stgcn_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ST-GCN specific")
        parser.add_argument('--edge_importance_weighting',
                            action='store_true',
                            help="Add a learnable multiplicative edge "
                            "importance weighting to the adjacency matrix.")
        parser.add_argument(
            '--partition_strategy',
            type=str,
            choices=["uniform", "distance", "spatial"],
            default="uniform",
            help="Partition strategy used to partition node neighbour sets.")
        parser.add_argument('--full_model',
                            action='store_true',
                            help="Train the full model as presented in the "
                            "paper instead of a more lightweight version.")
        return parent_parser

    def __init__(self,
                 num_classes,
                 partition_strategy="uniform",
                 edge_importance_weighting=False,
                 full_model=False,
                 **kwargs):
        super().__init__()

        graph = {
            "graph_layout":
            KinectV2,
            "graph_options":
            ST_GCN_Options(partition_strategy=partition_strategy,
                           edge_importance_weighting=edge_importance_weighting)
        }

        self.data_batch_norm = ChannelwiseBatchNorm(in_channels=3,
                                                    landmarks=25)
        self.person2batch = Person2Batch(person_dimension=1, num_persons=2)

        temporal_kernel_size = 9
        module_list = [
            SpatioTemporalGraphConvolution(
                in_channels=3,
                out_channels=64,
                temporal_kernel_size=temporal_kernel_size,
                temporal_stride=1,
                residual=False,
                **graph)
        ]
        if full_model:
            module_list += [
                SpatioTemporalGraphConvolution(
                    in_channels=64,
                    out_channels=64,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_stride=1,
                    **graph),
                SpatioTemporalGraphConvolution(
                    in_channels=64,
                    out_channels=64,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_stride=1,
                    **graph),
                SpatioTemporalGraphConvolution(
                    in_channels=64,
                    out_channels=64,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_stride=1,
                    **graph)
            ]
        module_list += [
            SpatioTemporalGraphConvolution(
                in_channels=64,
                out_channels=128,
                temporal_kernel_size=temporal_kernel_size,
                temporal_stride=2,
                **graph)
        ]
        if full_model:
            module_list += [
                SpatioTemporalGraphConvolution(
                    in_channels=128,
                    out_channels=128,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_stride=1,
                    **graph),
                SpatioTemporalGraphConvolution(
                    in_channels=128,
                    out_channels=128,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_stride=1,
                    **graph)
            ]
        module_list += [
            SpatioTemporalGraphConvolution(
                in_channels=128,
                out_channels=256,
                temporal_kernel_size=temporal_kernel_size,
                temporal_stride=2,
                **graph)
        ]
        if full_model:
            module_list += [
                SpatioTemporalGraphConvolution(
                    in_channels=256,
                    out_channels=256,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_stride=1,
                    **graph)
            ]
        module_list += [
            SpatioTemporalGraphConvolution(
                in_channels=256,
                out_channels=256,
                temporal_kernel_size=temporal_kernel_size,
                temporal_stride=1,
                **graph)
        ]
        self.st_gcn_networks = torch.nn.ModuleList(module_list)

        self.fully_connected = torch.nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Move persons into the batch dimension
        x = self.person2batch(x)

        # Normalise data
        x = self.data_batch_norm(x)

        # Forwad
        for gcn in self.st_gcn_networks:
            x = gcn(x)

        # Global pooling
        x = torch.nn.functional.avg_pool2d(x, x.size()[2:])

        # Aggregate results for people of each batch element
        x = self.person2batch.extract_persons(x)

        # Predict
        x = self.fully_connected(x)
        x = x.view(x.size(0), -1)

        return x
