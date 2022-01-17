import torch
from shar.datatransforms import Person2Batch
from shar.normalisations import ChannelwiseBatchNorm
from shar.graphs import SpatioTemporalGraphConvolution
from shar.transformers import SpatialTransformer, TemporalTransformer
from shar.graphs.graphlayouts import KinectV2
from shar.graphs.graphoptions import ST_GCN_Options

DEFAULT_NUM_LAYERS = 5


class STTR(torch.nn.Module):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("ST-TR specific argumetns")
        parser.add_argument(
            '--layers',
            type=int,
            choices=list(range(4, 10)),
            default=DEFAULT_NUM_LAYERS,
            help="Number of GCN & Transformer layers to use. This allows to "
            "train more lightweight models than the original paper (layers "
            "1-3 are GCN layers, all subsequent layers are transformers; "
            "layers=9 is as presented in the paper.)")
        return parent_parser

    def __init__(self,
                 keypoint_dim,
                 num_keypoints,
                 num_classes,
                 num_persons,
                 layers=DEFAULT_NUM_LAYERS,
                 **kwargs):
        super().__init__()

        graph = {
            "graph_layout": KinectV2,
            "graph_options": ST_GCN_Options(edge_importance_weighting=True)
        }

        self.data_batch_norm = ChannelwiseBatchNorm(in_channels=3,
                                                    landmarks=25)
        self.person2batch = Person2Batch(person_dimension=1, num_persons=2)

        temporal_kernel_size = 9
        # Define output_channels for all layers, first item is the input dim of
        # the very first layer. This in particular makes channels[layers] the
        # output dim of the last gcn layer
        channels = [keypoint_dim, 64, 64, 64, 128, 128, 128, 256, 256, 256]
        module_list = []
        for i in range(3):
            module_list += [
                SpatioTemporalGraphConvolution(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    temporal_kernel_size=temporal_kernel_size,
                    temporal_stride=1,
                    residual=False,
                    **graph)
            ]
        for i in range(3, layers):
            module_list += [
                SpatialTransformer(in_channels=channels[i],
                                   out_channels=channels[i + 1],
                                   dim_key=32,
                                   dim_value=32,
                                   num_heads=8,
                                   normalisation='batch'),
                TemporalTransformer(in_channels=channels[i + 1],
                                    out_channels=channels[i + 1],
                                    dim_key=32,
                                    dim_value=32,
                                    num_heads=8,
                                    normalisation='batch')
            ]
        self.layers = torch.nn.ModuleList(module_list)

        self.fully_connected = torch.nn.Conv2d(channels[layers],
                                               num_classes,
                                               kernel_size=1)

    def forward(self, x):
        # Move persons into the batch dimension
        x = self.person2batch(x)

        # Normalise data
        x = self.data_batch_norm(x)

        # Forwad
        for layer in self.layers:
            x = layer(x)

        # Global pooling
        x = torch.nn.functional.avg_pool2d(x, x.size()[2:])

        # Aggregate results for people of each batch element
        x = self.person2batch.extract_persons(x)

        # Predict
        x = self.fully_connected(x)
        x = x.view(x.size(0), -1)

        return x
