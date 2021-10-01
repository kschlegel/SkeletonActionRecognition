import argparse

import pytorch_lightning as pl

from datamodule import SkeletonDataModule
from models.actionrecognitionmodule import ActionRecognitionModule  # type: ignore
from models.stgcn import STGCN  # type: ignore
from models.agcn import AGCN  # type: ignore
from models.sttr import STTR  # type: ignore
from models.gcnlogsigrnn import GCNLOGSIG
from models.logsigrnn import LogSigRNNModel


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument('--model_name',
                        type=str,
                        choices=["stgcn", "agcn", "sttr", "gcnlogsigrnn", "logsigrnn"],
                        default='stgcn',
                        help='The model to train (default is stgcn)')

    parser = SkeletonDataModule.add_data_specific_args(parser)
    parser = ActionRecognitionModule.add_model_specific_args(parser)
    # -------
    temp_args, _ = parser.parse_known_args()
    if temp_args.model_name == "stgcn":
        parser = STGCN.add_stgcn_specific_args(parser)
    elif temp_args.model_name == "agcn":
        parser = AGCN.add_agcn_specific_args(parser)
    elif temp_args.model_name == "logsigrnn":
        parser = LogSigRNNModel.add_logsigrnn_specific_args(parser)

    # SET CUSTOM DEFAULTS (for convenience so I don't have to specify them)
    parser.set_defaults(gpus=1)
    parser.set_defaults(max_epochs=10)

    return parser.parse_args()


def main(hparams):
    hparams_dict = vars(hparams)

    data = SkeletonDataModule(**hparams_dict)

    if hparams.model_name == "stgcn":
        model = STGCN(num_classes=data.num_actions, **hparams_dict)
    elif hparams.model_name == "agcn":
        model = AGCN(num_classes=data.num_actions, **hparams_dict)
    elif hparams.model_name == "sttr":
        model = STTR(num_classes=data.num_actions, **hparams_dict)
    elif hparams.model_name == "gcnlogsigrnn":
        model = GCNLOGSIG(num_classes=data.num_actions, **hparams_dict)
    elif hparams.model_name == "logsigrnn":
        model = LogSigRNNModel(num_classes=data.num_actions, **hparams_dict)

    training_module = ActionRecognitionModule(model=model,
                                              num_classes=data.num_actions,
                                              **hparams_dict)

    trainer = pl.Trainer.from_argparse_args(hparams)
    trainer.fit(training_module, data)


if __name__ == '__main__':
    hparams = parse_arguments()
    main(hparams)
