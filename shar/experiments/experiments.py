from typing import Optional, Union, List
import argparse
import os
import sys
import importlib

import pytorch_lightning as pl
from .datamodule import SkeletonDataModule
from .actionrecognitionmodule import ActionRecognitionModule

from shar._utils.argparser import WithDefaultsWrapper


class Experiments:
    """
    Run standard SHAR experiments with an extensive command line interface.

    This class provides an way to run standard experients on different shar
    models with minimal effort, with an extensive list of command line options
    to customise individual runs.

    To add models for experimentation add one or more folders as the
    constructor argument. All python file in a given path that are not private
    (start with an underscore '_') are assumed to contain a model with the
    exact same name, up to capitalisations (e.g. a model LogSigRNN in the file
    logsigrnn.py, any naming so that class_name.lower() == "filename_without
    .py" is valid)
    For any model added this way if the model class contains a method
    'add_argparse_args(parser)' adding command line arguments to the given
    parser, these options will automatically be added to the Experiments
    command line interface. Moreover, a number of arguments to select and
    configure the data, optimizer etc used for training are added. Lastly, all
    PyTorch Lightning command line options are exposed.

    To use this class to run experiments all you need is the following code:
        from shar.experiments import Experiments
        exp = Experiments(PATH_TO_YOUR_MODELS)
        exp.run()

    """

    DataModuleClass = SkeletonDataModule
    TrainingModuleClass = ActionRecognitionModule

    def __init__(self,
                 model_dirs: Optional[Union[str, List[str]]] = None) -> None:
        """
        Parameters
        ----------
        model_dirs : str or list of strings, optional (default is './models')
            Path(s) where the implementations for models can be found. Every
            python file with a filename not starting with an underscore in a
            given directory is assumed to implement a model.
        """
        self._models = {}
        if model_dirs is None:
            model_dirs = []
        if isinstance(model_dirs, str):
            model_dirs = [model_dirs]
        model_dirs += [os.path.join(os.path.dirname(__file__), "models")]
        for model_dir in model_dirs:
            for filename in os.listdir(model_dir):
                if not filename.startswith("_") and filename.endswith(".py"):
                    model_name = filename[:-3]
                    spec = importlib.util.spec_from_file_location(
                        model_name, os.path.join(model_dir,
                                                 model_name + ".py"))
                    if spec is None:
                        raise Exception("Could not find model", model_name)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[model_name] = module
                    spec.loader.exec_module(module)  # type: ignore

                    for name in dir(sys.modules[model_name]):
                        if name.lower() == model_name:
                            self._models[model_name] = getattr(
                                sys.modules[model_name], name)

        self.parse_arguments()

    def parse_arguments(self) -> None:
        """
        Adds command line options for all experiment modules.

        This is called automatically by the constructor so should not need to
        be called manually.
        """
        argparser = argparse.ArgumentParser()
        argparser = pl.Trainer.add_argparse_args(argparser)

        parser = WithDefaultsWrapper(argparser)
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
        parser.add_argument('--random_seed',
                            type=int,
                            help="If given seeds (all) random number "
                            "generators with the given seed.")

        parser = self.DataModuleClass.add_data_specific_args(parser)
        parser = self.TrainingModuleClass.add_model_specific_args(parser)
        # -------
        model_name: Optional[str] = None
        for i, arg in enumerate(sys.argv):
            if arg == "--model_name":
                model_name = sys.argv[i + 1]
                break
        if model_name is not None:
            if hasattr(self._models[model_name], "add_argparse_args"):
                parser = self._models[model_name].add_argparse_args(parser)

        self._hparams = parser.parse_args()

    def run(self) -> None:
        """
        Run an experiment with the setting defined through the CLI.

        Does everything from setting up the data to running the experiment
        based on the command line options which have been established by the
        end of the constructor. Thus each call to run will run an identical
        (but independent from previous runs) experiment based on CLI options.
        """
        hparams_dict = vars(self._hparams)

        if self._hparams.random_seed is not None:
            pl.utilities.seed.seed_everything(seed=self._hparams.random_seed,
                                              workers=True)

        data = self.DataModuleClass(**hparams_dict)
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

        training_module = self.TrainingModuleClass(model=model, **hparams_dict)

        experiment_name = "lightning_logs"
        checkpoint_path = "checkpoints"
        if self._hparams.experiment_name is not None:
            experiment_name += "_" + self._hparams.experiment_name
            checkpoint_path += "_" + self._hparams.experiment_name
        if not os.path.exists(experiment_name):
            os.mkdir(experiment_name)
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)
        loggers: List[pl.loggers.base.LightningLoggerBase] = [
            pl.loggers.TensorBoardLogger(save_dir=".",
                                         name=experiment_name,
                                         default_hp_metric=False)
        ]
        save_dir = loggers[-1].save_dir
        if save_dir is None:
            save_dir = "."
        loggers += [
            pl.loggers.CSVLogger(save_dir=save_dir,
                                 name=experiment_name,
                                 version=loggers[-1].version)
        ]

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(checkpoint_path, "version_" +
                                 str(loggers[0].version)))

        trainer = pl.Trainer.from_argparse_args(
            self._hparams, logger=loggers, callbacks=[checkpoint_callback])
        trainer.fit(training_module, data)
