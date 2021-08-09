import torch
import pytorch_lightning as pl
from torchmetrics import Accuracy, AveragePrecision

DEFAULT_LR = 0.001


class ActionRecognitionModule(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group(
            "Training specific arguments")
        parser.add_argument(
            "--lr",
            type=float,
            default=DEFAULT_LR,
            help="Learning rate (default is {})".format(DEFAULT_LR))
        return parent_parser

    def __init__(self, model, num_classes, lr=DEFAULT_LR, **kwargs):
        super().__init__()
        # log module specific command line arguments
        self.save_hyperparameters('num_classes', 'lr')
        # log model class
        self.save_hyperparameters({"model": model.__class__})
        # log all remaining, not module specific command line arguments
        self.save_hyperparameters(kwargs)

        self.model = model

        self.loss = torch.nn.CrossEntropyLoss()

        self.accuracy = Accuracy()
        self.average_precision = AveragePrecision(num_classes=num_classes,
                                                  compute_on_step=False)

    def forward(self, x):
        predictions = self.model(x)
        predictions = torch.argmax(predictions, dim=1)
        return predictions

    # ##### TRAINING #####

    def training_step(self, batch, batch_idx):
        keypoints, labels = batch

        predictions = self.model(keypoints)
        loss = self.loss(predictions, labels)

        self.log("loss/train", loss, on_epoch=True, on_step=True)
        return loss

    # ##### VALIDATION #####

    def validation_step(self, batch, batch_idx):
        keypoints, labels = batch

        predictions = self.model(keypoints)
        loss = self.loss(predictions, labels)
        self.log("loss/val", loss, on_epoch=True, on_step=False)

        predictions = torch.softmax(predictions, dim=1)
        self.accuracy(predictions, labels)
        self.log('metrics/accuracy',
                 self.accuracy,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
        self.average_precision(predictions, labels)

        return loss

    def validation_epoch_end(self, outs):
        aps = [
            v if not torch.isnan(v) else 0
            for v in self.average_precision.compute()
        ]
        mAP = torch.mean(torch.tensor(aps, device=self.device))
        self.log('metrics/mean_average_precision', mAP)
        self.log('hp_metric', mAP)

    # ##### OPTIMIZER #####

    def configure_optimizers(self):
        return torch.optim.Adam(self.classifier.parameters(),
                                lr=self.hparams.lr)
