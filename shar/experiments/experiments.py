from typing import Union, List
import argparse
import os
import sys
import importlib

import pytorch_lightning as pl
from .datamodule import SkeletonDataModule
from .actionrecognitionmodule import ActionRecognitionModule

from shar._utils.argparser import WithDefaultsWrapper


class Experiments:
    def __init__(self, model_dirs: Union[str, List[str]] = "./models"):
        """
        Parameters
        ----------
        model_dirs : str or list of strings, optional (default is './models')
            Path(s) where the implementations for models can be found. Every
            python file with a filename not starting with an underscore in a
            given directory is assumed to implement a model.
        """
        self._models = {}
        if isinstance(model_dirs, str):
            model_dirs = [model_dirs]
        for model_dir in model_dirs:
            for filename in os.listdir(model_dir):
                if not filename.startswith("_") and filename.endswith(".py"):
                    model_name = filename[:-3]
                    spec = importlib.util.spec_from_file_location(
                        model_name, os.path.join(model_dir,
                                                 model_name + ".py"))
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[model_name] = module
                    spec.loader.exec_module(module)

                    for name in dir(sys.modules[model_name]):
                        if name.lower() == model_name:
                            self._models[model_name] = getattr(
                                sys.modules[model_name], name)

        self.parse_arguments()

    def parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser = pl.Trainer.add_argparse_args(parser)

        parser = WithDefaultsWrapper(parser)
        parser_group = parser.add_argument_group(
            "Experiments specific arguments)")
        parser_group.add_argument('--model_name',
                                  type=str,
                                  choices=self._models.keys(),
                                  help='The model to train')
        parser_group.add_argument(
            '--additional',
            nargs="*",
            type=str,
            help="Additional info to be logged about this run")
        parser_group.add_argument(
            '-x',
            '--experiment_name',
            type=str,
            help="Optional additional experiment identifier.")

        parser = SkeletonDataModule.add_data_specific_args(parser)
        parser = ActionRecognitionModule.add_model_specific_args(parser)
        # -------
        for i, arg in enumerate(sys.argv):
            if arg == "--model_name":
                model_name = sys.argv[i + 1]
                break
        else:
            model_name = None
        if model_name is not None:
            if hasattr(self._models[model_name], "add_argparse_args"):
                parser = self._models[model_name].add_argparse_args(parser)

        self._hparams = parser.parse_args()

    def run(self):
        hparams_dict = vars(self._hparams)

        data = SkeletonDataModule(**hparams_dict)
        if data.keypoint_dim is None or data.num_keypoints is None:
            # This happens when we load from file without datasetloader
            # specified. We then don't have apriori info about the data
            data.prepare_data()
            data.setup()
        hparams_dict["keypoint_dim"] = data.keypoint_dim
        hparams_dict["num_keypoints"] = data.num_keypoints
        hparams_dict["num_classes"] = data.num_actions
        hparams_dict["class_labels"] = data.class_labels

        if self._hparams.model_name in self._models:
            model = self._models[self._hparams.model_name](**hparams_dict)
        else:
            raise Exception("Invalid model name.")

        training_module = ActionRecognitionModule(model=model, **hparams_dict)

        experiment_name = "lightning_logs"
        checkpoint_path = "checkpoints"
        if self._hparams.experiment_name is not None:
            experiment_name += "_" + self._hparams.experiment_name
            checkpoint_path += "_" + self._hparams.experiment_name
        if not os.path.exists(experiment_name):
            os.mkdir(experiment_name)
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        loggers = [
            pl.loggers.TensorBoardLogger(save_dir=".",
                                         name=experiment_name,
                                         default_hp_metric=False)
        ]
        loggers += [
            pl.loggers.CSVLogger(save_dir=loggers[-1].save_dir,
                                 name=experiment_name,
                                 version=loggers[-1].version)
        ]

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(checkpoint_path, "version_" +
                                 str(loggers[0].version)))

        trainer = pl.Trainer.from_argparse_args(
            self._hparams, logger=loggers, callbacks=[checkpoint_callback])
        trainer.fit(training_module, data)
