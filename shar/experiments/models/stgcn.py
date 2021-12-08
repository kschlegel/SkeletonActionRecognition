import torch
from shar.datatransforms import Person2Batch
from shar.normalisations import ChannelwiseBatchNorm
from shar.graphs import SpatioTemporalGraphConvolution


class STGCN(torch.nn.Module):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("ST-GCN specific")
        parser.add_argument('--edge_importance_weighting',
                            action='store_true',
                            help="Add a learnable multiplicative edge "
                            "importance weighting to the adjacency matrix.")
        parser.add_argument('--full_model',
                            action='store_true',
                            help="Train the full model as presented in the "
                            "paper instead of a more lightweight version.")
        return parent_parser

    def __init__(self,
                 keypoint_dim,
                 num_keypoints,
                 num_classes,
                 num_persons,
                 edge_importance_weighting=False,
                 full_model=False,
                 **kwargs):
        super().__init__()

        graph_options = {
            "edge_importance_weighting": edge_importance_weighting,
        }

        self.data_batch_norm = ChannelwiseBatchNorm(in_channels=keypoint_dim,
                                                    landmarks=num_keypoints)
        self.person2batch = Person2Batch(person_dimension=1,
                                         num_persons=num_persons)

        temporal_kernel_size = 9
        module_list = [
            SpatioTemporalGraphConvolution(
                graph_options=graph_options,
                in_channels=keypoint_dim,
                out_channels=64,
                temporal_kernel_size=temporal_kernel_size,
                temporal_stride=1,
                residual=False)
        ]
        if full_model:
            module_list += [
                SpatioTemporalGraphConvolution(
                    graph_options=graph_options,
                    in_channels=64,
                    out_channels=64,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_stride=1),
                SpatioTemporalGraphConvolution(
                    graph_options=graph_options,
                    in_channels=64,
                    out_channels=64,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_stride=1),
                SpatioTemporalGraphConvolution(
                    graph_options=graph_options,
                    in_channels=64,
                    out_channels=64,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_stride=1)
            ]
        module_list += [
            SpatioTemporalGraphConvolution(
                graph_options=graph_options,
                in_channels=64,
                out_channels=128,
                temporal_kernel_size=temporal_kernel_size,
                temporal_stride=2)
        ]
        if full_model:
            module_list += [
                SpatioTemporalGraphConvolution(
                    graph_options=graph_options,
                    in_channels=128,
                    out_channels=128,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_stride=1),
                SpatioTemporalGraphConvolution(
                    graph_options=graph_options,
                    in_channels=128,
                    out_channels=128,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_stride=1)
            ]
        module_list += [
            SpatioTemporalGraphConvolution(
                graph_options=graph_options,
                in_channels=128,
                out_channels=256,
                temporal_kernel_size=temporal_kernel_size,
                temporal_stride=2)
        ]
        if full_model:
            module_list += [
                SpatioTemporalGraphConvolution(
                    graph_options=graph_options,
                    in_channels=256,
                    out_channels=256,
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_stride=1)
            ]
        module_list += [
            SpatioTemporalGraphConvolution(
                graph_options=graph_options,
                in_channels=256,
                out_channels=256,
                temporal_kernel_size=temporal_kernel_size,
                temporal_stride=1)
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
