import torch
from skeletonactionrecognition.datatransforms import Person2Batch
from skeletonactionrecognition.normalisations import ChannelwiseBatchNorm
from skeletonactionrecognition.graphs import SpatioTemporalGraphConvolution


class AGCN(torch.nn.Module):
    @staticmethod
    def add_agcn_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("AGCN specific")
        parser.add_argument('--no_learnable_adjacency',
                            action='store_true',
                            help="Add a learnable component to the adjacency "
                            "matrix.")
        parser.add_argument('--no_data_dependent_adjacency',
                            action='store_true',
                            help="Add a data dependent component to the "
                            "adjacency matrix.")
        return parent_parser

    def __init__(self,
                 num_classes,
                 no_learnable_adjacency=False,
                 no_data_dependent_adjacency=False,
                 **kwargs):
        super().__init__()

        graph_options = {
            "learnable_adjacency": not no_learnable_adjacency,
            "data_dependent_adjacency": not no_data_dependent_adjacency
        }

        self.data_batch_norm = ChannelwiseBatchNorm(in_channels=3,
                                                    landmarks=25)
        self.person2batch = Person2Batch(person_dimension=1, num_persons=2)

        temporal_kernel_size = 9
        self.st_gcn_networks = torch.nn.ModuleList((
            SpatioTemporalGraphConvolution(
                graph_options=graph_options,
                in_channels=3,
                out_channels=64,
                temporal_kernel_size=temporal_kernel_size,
                temporal_stride=1,
                residual=False),
            # SpatioTemporalGraphConvolution(
            #     graph_options=graph_options,
            #     in_channels=64,
            #     out_channels=64,
            #     temporal_kernel_size=temporal_kernel_size,
            #     temporal_stride=1,
            #     residual=False),
            # SpatioTemporalGraphConvolution(
            #     graph_options=graph_options,
            #     in_channels=64,
            #     out_channels=64,
            #     temporal_kernel_size=temporal_kernel_size,
            #     temporal_stride=1,
            #     residual=False),
            # SpatioTemporalGraphConvolution(
            #     graph_options=graph_options,
            #     in_channels=64,
            #     out_channels=64,
            #     temporal_kernel_size=temporal_kernel_size,
            #     temporal_stride=1,
            #     residual=False),
            SpatioTemporalGraphConvolution(
                graph_options=graph_options,
                in_channels=64,
                out_channels=128,
                temporal_kernel_size=temporal_kernel_size,
                temporal_stride=2,
                residual=False),
            # SpatioTemporalGraphConvolution(
            #     graph_options=graph_options,
            #     in_channels=128,
            #     out_channels=128,
            #     temporal_kernel_size=temporal_kernel_size,
            #     temporal_stride=1,
            #     residual=False),
            # SpatioTemporalGraphConvolution(
            #     graph_options=graph_options,
            #     in_channels=128,
            #     out_channels=128,
            #     temporal_kernel_size=temporal_kernel_size,
            #     temporal_stride=1,
            #     residual=False),
            SpatioTemporalGraphConvolution(
                graph_options=graph_options,
                in_channels=128,
                out_channels=256,
                temporal_kernel_size=temporal_kernel_size,
                temporal_stride=2,
                residual=False),
            # SpatioTemporalGraphConvolution(
            #     graph_options=graph_options,
            #     in_channels=256,
            #     out_channels=256,
            #     temporal_kernel_size=temporal_kernel_size,
            #     temporal_stride=1,
            #     residual=False),
            SpatioTemporalGraphConvolution(
                graph_options=graph_options,
                in_channels=256,
                out_channels=256,
                temporal_kernel_size=temporal_kernel_size,
                temporal_stride=1,
                residual=False),
        ))

        self.fully_connected = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        # Move persons into the batch dimension
        x = self.person2batch(x)

        # Normalise data
        x = self.data_batch_norm(x)

        # Forwad
        for gcn in self.st_gcn_networks:
            x = gcn(x)

        # average results accross nodes and remaining frames
        x = x.view(x.shape[:2] +
                   (-1, ))  # (batch*person, channels, frame*node)
        x = x.mean(2)

        # Aggregate results for people of each batch element
        x = self.person2batch.extract_persons(x)
        # Predict
        x = self.fully_connected(x)
        return x
