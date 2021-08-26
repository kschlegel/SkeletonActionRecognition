import torch
from shar.datatransforms import Person2Batch
from shar.normalisations import ChannelwiseBatchNorm
from shar.graphs import SpatioTemporalGraphConvolution
from shar.transformers import SpatialTransformer, TemporalTransformer


class STTR(torch.nn.Module):
    @staticmethod
    def add_sttr_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("ST-TR specific")
        parser.add_argument('--full_model',
                            action='store_true',
                            help="Train the full model as presented in the "
                            "paper instead of a more lightweight version.")
        return parent_parser

    def __init__(self, num_classes, full_model=False, **kwargs):
        super().__init__()

        graph_options = {
            "edge_importance_weighting": True,
        }

        self.data_batch_norm = ChannelwiseBatchNorm(in_channels=3,
                                                    landmarks=25)
        self.person2batch = Person2Batch(person_dimension=1, num_persons=2)

        temporal_kernel_size = 9
        module_list = [
            SpatioTemporalGraphConvolution(
                graph_options=graph_options,
                in_channels=3,
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
            ]
        module_list += [
            SpatialTransformer(in_channels=64,
                               out_channels=128,
                               dim_key=32,
                               dim_value=32,
                               num_heads=8,
                               normalisation='batch'),
            TemporalTransformer(in_channels=128,
                                out_channels=128,
                                dim_key=32,
                                dim_value=32,
                                num_heads=8,
                                normalisation='batch')
        ]
        if full_model:
            module_list += [
                SpatialTransformer(in_channels=128,
                                   out_channels=128,
                                   dim_key=32,
                                   dim_value=32,
                                   num_heads=8,
                                   normalisation='batch'),
                TemporalTransformer(in_channels=128,
                                    out_channels=128,
                                    dim_key=32,
                                    dim_value=32,
                                    num_heads=8,
                                    normalisation='batch'),
                SpatialTransformer(in_channels=128,
                                   out_channels=128,
                                   dim_key=32,
                                   dim_value=32,
                                   num_heads=8,
                                   normalisation='batch'),
                TemporalTransformer(in_channels=128,
                                    out_channels=128,
                                    dim_key=32,
                                    dim_value=32,
                                    num_heads=8,
                                    normalisation='batch')
            ]
        module_list += [
            SpatialTransformer(in_channels=128,
                               out_channels=256,
                               dim_key=64,
                               dim_value=64,
                               num_heads=8,
                               normalisation='batch'),
            TemporalTransformer(in_channels=256,
                                out_channels=256,
                                dim_key=64,
                                dim_value=64,
                                num_heads=8,
                                normalisation='batch')
        ]
        if full_model:
            module_list += [
                SpatialTransformer(in_channels=256,
                                   out_channels=256,
                                   dim_key=64,
                                   dim_value=64,
                                   num_heads=8,
                                   normalisation='batch'),
                TemporalTransformer(in_channels=256,
                                    out_channels=256,
                                    dim_key=64,
                                    dim_value=64,
                                    num_heads=8,
                                    normalisation='batch')
            ]
        module_list += [
            SpatialTransformer(in_channels=256,
                               out_channels=256,
                               dim_key=64,
                               dim_value=64,
                               num_heads=8,
                               normalisation='batch'),
            TemporalTransformer(in_channels=256,
                                out_channels=256,
                                dim_key=64,
                                dim_value=64,
                                num_heads=8,
                                normalisation='batch')
        ]
        self.sub_networks = torch.nn.ModuleList(module_list)

        self.fully_connected = torch.nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # Move persons into the batch dimension
        x = self.person2batch(x)

        # Normalise data
        x = self.data_batch_norm(x)

        # Forwad
        for n in self.sub_networks:
            x = n(x)

        # Global pooling
        x = torch.nn.functional.avg_pool2d(x, x.size()[2:])

        # Aggregate results for people of each batch element
        x = self.person2batch.extract_persons(x)

        # Predict
        x = self.fully_connected(x)
        x = x.view(x.size(0), -1)

        return x
