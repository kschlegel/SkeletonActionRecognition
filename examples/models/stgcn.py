import torch
from skeletonactionrecognition.normalisations import ChannelwiseBatchNorm
from skeletonactionrecognition.graphs import SpatioTemporalGraphConvolution


class STGCN(torch.nn.Module):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ST-GCN specific")
        parser.add_argument('--edge_importance_weighting',
                            action='store_true',
                            help="Add a learnable multiplicative edge "
                            "importance weighting to the adjacency matrix.")
        return parent_parser

    def __init__(self, num_classes, edge_importance_weighting=False, **kwargs):
        super().__init__()

        graph_options = {
            "edge_importance_weighting": edge_importance_weighting,
        }

        self.data_batch_norm = ChannelwiseBatchNorm(in_channels=3,
                                                    landmarks=25)

        temporal_kernel_size = 9
        self.st_gcn_networks = torch.nn.ModuleList((
            SpatioTemporalGraphConvolution(
                graph_options=graph_options,
                in_channels=3,
                out_channels=64,
                temporal_kernel_size=temporal_kernel_size,
                temporal_stride=1,
                residual=False),
            #SpatioTemporalGraphConvolution(graph=self.graph, in_channels=64, out_channels=64, temporal_kernel_size=temporal_kernel_size,  temporal_stride=1),
            #SpatioTemporalGraphConvolution(graph=self.graph, in_channels=64, out_channels=64, temporal_kernel_size=temporal_kernel_size,  temporal_stride=1),
            #SpatioTemporalGraphConvolution(graph=self.graph, in_channels=64, out_channels=64, temporal_kernel_size=temporal_kernel_size,  temporal_stride=1),
            SpatioTemporalGraphConvolution(
                graph_options=graph_options,
                in_channels=64,
                out_channels=128,
                temporal_kernel_size=temporal_kernel_size,
                temporal_stride=2),
            #SpatioTemporalGraphConvolution(graph=self.graph, in_channels=128, out_channels=128, temporal_kernel_size=temporal_kernel_size, temporal_stride=1),
            #SpatioTemporalGraphConvolution(graph=self.graph, in_channels=128, out_channels=128, temporal_kernel_size=temporal_kernel_size, temporal_stride=1),
            SpatioTemporalGraphConvolution(
                graph_options=graph_options,
                in_channels=128,
                out_channels=256,
                temporal_kernel_size=temporal_kernel_size,
                temporal_stride=2),
            #SpatioTemporalGraphConvolution(graph=self.graph, in_channels=256, out_channels=256, temporal_kernel_size=temporal_kernel_size, temporal_stride=1),
            SpatioTemporalGraphConvolution(
                graph_options=graph_options,
                in_channels=256,
                out_channels=256,
                temporal_kernel_size=temporal_kernel_size,
                temporal_stride=1),
        ))

        self.fully_connected = torch.nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # TODO: with NTU I'll have a person dimension here to reduce
        batch, channels, frames, nodes = x.size()

        # data normalization
        x = self.data_batch_norm(x)

        # forwad
        for gcn in self.st_gcn_networks:
            x = gcn(x)

        # global pooling
        x = torch.nn.functional.avg_pool2d(x, x.size()[2:])
        x = x.view(batch, -1, 1, 1)

        # prediction
        x = self.fully_connected(x)
        x = x.view(x.size(0), -1)

        return x
