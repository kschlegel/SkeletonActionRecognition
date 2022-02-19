import torch

from shar.datatransforms import Person2Batch


class HCN(torch.nn.Module):
    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("HCN specific arguments")
        parser.add_argument('--small',
                            action='store_true',
                            help="Use the smaller version of the HCN network.")
        return parent_parser

    def __init__(self,
                 keypoint_dim,
                 num_keypoints,
                 num_classes,
                 num_persons,
                 target_len,
                 small=False,
                 **kwargs):
        # target_len is used by SkeletonDataset to normalise sequences to fixed
        # len, we pick that up here to know how many frames to expect
        super().__init__()
        if small:
            divisor = 4
        else:
            divisor = 1

        self.person2batch = Person2Batch(person_dimension=1,
                                         num_persons=2,
                                         aggregation="max")

        self.backbone = BackBone(keypoint_dim, num_keypoints, small=small)
        self.conv6 = torch.nn.Conv2d(in_channels=128 // divisor,
                                     out_channels=256 // divisor,
                                     kernel_size=(3, 3),
                                     padding=(1, 1))
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))

        if small:
            # 64 * 2/8 * F = 16 * F
            in_features = 16 * target_len
        else:
            # 256 * 2/16 * F = 32 * F
            in_features = 32 * target_len
        self.linear7 = torch.nn.Linear(in_features=in_features,
                                       out_features=256 // divisor)
        self.linear8 = torch.nn.Linear(in_features=256 // divisor,
                                       out_features=num_classes)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        # x = (B, F, L, D)
        # Move persons into the batch dimension
        x = self.person2batch(x)

        x = self.backbone(x)
        # => (B, 128, F/8, 4)
        # => (B, 128|32, F/8|4, 4)
        x = self.conv6(x)
        # => (B, 256|64, F/8|4, 4)
        x = self.pool(x)
        # => (B, 256|64, F/16|8, 2)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)

        # Aggregate results for people of each batch element
        x = self.person2batch.extract_persons(x)

        x = x.view(x.shape[0], -1)
        # => (B, 256 * F/16 * 2) = (B, 32 * F)
        # => (B, 64 * F/8 * 2) = (B, 16 * F)
        x = self.linear7(x)
        # => (B, 256|54)
        x = self.dropout(x)
        x = torch.nn.functional.relu(x)
        x = self.linear8(x)
        return x


class FeatureBranch(torch.nn.Module):
    def __init__(self, keypoint_dim, num_keypoints, small=False):
        super().__init__()
        self._small = small
        if small:
            divisor = 2
        else:
            divisor = 1
        self.conv1 = torch.nn.Conv2d(in_channels=keypoint_dim,
                                     out_channels=64 // divisor,
                                     kernel_size=(1, 1))
        self.conv2 = torch.nn.Conv2d(in_channels=64 // divisor,
                                     out_channels=32 // divisor,
                                     kernel_size=(3, 1),
                                     padding=(1, 0))
        self.conv3 = torch.nn.Conv2d(in_channels=num_keypoints,
                                     out_channels=32 // divisor,
                                     kernel_size=(3, 3),
                                     padding=(1, 1))
        if not small:
            self.conv4 = torch.nn.Conv2d(in_channels=32,
                                         out_channels=64,
                                         kernel_size=(3, 3),
                                         padding=(1, 1))
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        # x = (B, D, F, L)
        x = self.conv1(x)
        # => (B, 64|32, F, L)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        # => (B, 32|16, F, L)
        x = torch.transpose(x, 1, 3)
        # => (B, L, F, 32|16)
        x = self.conv3(x)
        # => (B, 32|16, F, 32|16)
        x = self.pool(x)
        # => (B, 32|16, F/2, 16|8)
        if not self._small:
            x = self.conv4(x)
            # => (B, 64, F/2, 16)
            x = self.pool(x)
            # => (B, 64, F/4, 8)
        x = self.dropout(x)
        # => (B, 64|16, F/4|2, 8)
        return x


class BackBone(torch.nn.Module):
    def __init__(self, keypoint_dim, num_keypoints, small=False):
        super().__init__()
        if small:
            divisor = 4
        else:
            divisor = 1

        self.point_features = FeatureBranch(keypoint_dim,
                                            num_keypoints,
                                            small=small)
        self.motion_features = FeatureBranch(keypoint_dim,
                                             num_keypoints,
                                             small=small)

        self.conv5 = torch.nn.Conv2d(in_channels=128 // divisor,
                                     out_channels=128 // divisor,
                                     kernel_size=(3, 3),
                                     padding=(1, 1))
        self.pool = torch.nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x):
        # x = (B, F, L, D)
        zeros = torch.zeros((x.shape[0], x.shape[2], x.shape[3]),
                            device=x.device,
                            dtype=x.dtype)
        motion = torch.stack(
            [zeros] +
            [torch.sub(x[:, t], x[:, t - 1]) for t in range(1, x.shape[1])],
            dim=1)
        # => (B, F, L, D)
        pf = self.point_features(x)
        # => (B, 64|16, F/4|2, 8)
        mf = self.motion_features(motion)
        # =>(B, 64|16, F/4|2, 8)
        x = torch.cat((pf, mf), dim=1)
        # => (B, 128|32, F/4|2, 8)
        x = self.conv5(x)
        # => (B, 128|32, F/4|2, 8)
        x = self.pool(x)
        # => (B, 128|32, F/8|4, 4)
        x = torch.nn.functional.relu(x)
        x = self.dropout(x)
        return x
