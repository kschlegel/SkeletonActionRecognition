import torch
from shar.datatransforms import Person2Batch
from shar.normalisations import ChannelwiseBatchNorm
from shar.signatures import LogSigRNN


class LogSigRNNModel(torch.nn.Module):
    @staticmethod
    def add_logsigrnn_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LogSigRNN specific")
        parser.add_argument('--num_segments', type=int, default=50,
                            help="Number of segments")
        parser.add_argument('--lstm_channels', type=int, default=96,
                            help="LSTM output channels")
        return parent_parser

    def __init__(self,
                 num_classes,
                 num_segments=50,
                 lstm_channels=96,
                 **kwargs):
        super().__init__()

        self.data_batch_norm = ChannelwiseBatchNorm(in_channels=3,
                                                    landmarks=25)
        self.person2batch = Person2Batch(person_dimension=1, num_persons=2)

        self.logsigrnn = LogSigRNN(
            in_channels=3,
            logsignature_lvl=2,
            num_segments=num_segments,
            out_channels=lstm_channels
        )

        self.fully_connected = torch.nn.Linear(lstm_channels, num_classes)

    def forward(self, x):
        # Move persons into the batch dimension
        x = self.person2batch(x)
        # Normalise data
        x = self.data_batch_norm(x)

        x = self.logsigrnn(x)

        batch, out_channels = x.shape[:2]

        x = x.reshape(batch, out_channels, -1)
        x = x.mean(2)

        x = self.person2batch.extract_persons(x)

        # Predict
        x = self.fully_connected(x)
        x = x.view(x.size(0), -1)

        return x
