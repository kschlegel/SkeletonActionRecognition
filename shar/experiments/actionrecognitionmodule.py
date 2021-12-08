import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy, AveragePrecision, ConfusionMatrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from shar._utils.argparser import WithDefaultsWrapper

DEFAULT_LR = 0.001
DEFAULT_OPTIMIZER = "adam"


class ActionRecognitionModule(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        if isinstance(parent_parser, WithDefaultsWrapper):
            local_parser = parent_parser
        else:
            local_parser = WithDefaultsWrapper(parent_parser)
        parser = local_parser.add_argument_group("Training specific arguments")
        parser.add_argument("--lr",
                            type=float,
                            default=DEFAULT_LR,
                            help="Learning rate")
        parser.add_argument('--optimizer',
                            type=str,
                            choices=["adam", "sgd"],
                            default=DEFAULT_OPTIMIZER,
                            help="Optimizer to use for training.")
        parser.add_argument('--mAP',
                            action="store_true",
                            help="Compute mean average precision metric")
        parser.add_argument('--confusion_matrix',
                            action="store_true",
                            help="Log validation confusion matrix in "
                            "TensorBoard at the end of each epoch")
        parser.add_argument(
            '--training_metrics',
            action="store_true",
            help="Compute and log metrics also for training steps.")
        parser.add_argument(
            '--parameter_histograms',
            action="store_true",
            help="Plot histograms of all parameters and their gradients")

        return parent_parser

    def __init__(self,
                 model,
                 num_classes,
                 lr=DEFAULT_LR,
                 class_labels=None,
                 mAP=False,
                 confusion_matrix=False,
                 training_metrics=False,
                 parameter_histograms=False,
                 **kwargs):
        super().__init__()
        # log module specific command line arguments
        self.save_hyperparameters('num_classes', 'lr')
        # log model class
        self.save_hyperparameters(
            {"model": str(model.__class__).split(".")[-1][:-2]})
        self.save_hyperparameters(
            {"parameters": sum(p.numel() for p in model.parameters())})
        self.save_hyperparameters({
            "trainable parameters":
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        })
        # log all remaining, not module specific command line arguments
        self.save_hyperparameters(kwargs)

        self.model = model

        self.loss = torch.nn.CrossEntropyLoss()

        self.training_metrics = training_metrics
        self.accuracy = Accuracy()
        if training_metrics:
            self.train_accuracy = Accuracy()
        if mAP:
            self.mAP = AveragePrecision(num_classes=num_classes)
            if training_metrics:
                self.train_mAP = AveragePrecision(num_classes=num_classes)
        else:
            self.mAP = None

        if confusion_matrix:
            self.confusion_matrix = ConfusionMatrix(num_classes=num_classes,
                                                    compute_on_step=False,
                                                    normalize="true")
            self._class_labels = class_labels
        else:
            self.confusion_matrix = None

        self._parameter_histograms = parameter_histograms

    def forward(self, x):
        predictions = self.model(x)
        predictions = torch.argmax(predictions, dim=1)
        return predictions

    def setup(self, stage):
        if isinstance(self.logger, pl.loggers.base.LoggerCollection):
            for logger in self.logger:
                if isinstance(logger, pl.loggers.TensorBoardLogger):
                    self._tb_logger = logger
        else:
            self._tb_logger = self.logger

    # ##### TRAINING #####

    def on_train_start(self):
        metrics = {"metrics/acc": 0}
        if self.mAP is not None:
            metrics["metrics/mAP"] = 0
        self._tb_logger.log_hyperparams(self.hparams, metrics)

    def training_step(self, batch, batch_idx):
        keypoints, labels = batch

        predictions = self.model(keypoints)
        loss = self.loss(predictions, labels)
        self.log("loss/train", loss, on_epoch=True, on_step=True)

        if self.training_metrics:
            predictions = torch.softmax(predictions, dim=1)
            self.log('metrics/train_acc',
                     self.train_accuracy(predictions, labels),
                     on_step=False,
                     on_epoch=True)
            if self.mAP is not None:
                self.log('metrics/train_mAP',
                         self.train_mAP(predictions, labels),
                         on_step=False,
                         on_epoch=True)

        return loss

    def on_after_backward(self):
        if self._parameter_histograms:
            global_step = self.global_step
            for name, param in self.model.named_parameters():
                self.logger.experiment.add_histogram(name, param, global_step)
                if param.requires_grad:
                    self.logger.experiment.add_histogram(
                        f"{name}_grad", param.grad, global_step)

    # ##### VALIDATION #####

    def validation_step(self, batch, batch_idx):
        keypoints, labels = batch

        predictions = self.model(keypoints)
        loss = self.loss(predictions, labels)
        self.log("loss/val", loss, on_epoch=True, on_step=False)

        predictions = torch.softmax(predictions, dim=1)
        self.log('metrics/acc',
                 self.accuracy(predictions, labels),
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        if self.mAP is not None:
            self.log('metrics/mAP',
                     self.mAP(predictions, labels),
                     on_step=False,
                     on_epoch=True)
        if self.confusion_matrix is not None:
            self.confusion_matrix(predictions, labels)

        return loss

    def validation_epoch_end(self, outs):
        if self.confusion_matrix is not None:
            cm = self.confusion_matrix.compute()
            cm_display = ConfusionMatrixDisplay(
                cm.cpu().numpy(), display_labels=self._class_labels)
            fig, ax = plt.subplots(figsize=(10, 10))
            if self._class_labels is None:
                xticks_rotation = "horizontal"
            else:
                xticks_rotation = "vertical"
            cm_display.plot(ax=ax,
                            values_format=".2f",
                            xticks_rotation=xticks_rotation)
            self._tb_logger.experiment.add_figure(
                "Confusion matrix", fig, global_step=self.current_epoch)

    # ##### OPTIMIZER #####

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            return torch.optim.Adam(self.model.parameters(),
                                    lr=self.hparams.lr)
        elif self.hparams.optimizer == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=self.hparams.lr)
        else:
            raise Exception("Invalid optimizer")
