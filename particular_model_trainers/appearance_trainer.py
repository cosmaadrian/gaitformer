import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .acumen_trainer import AcumenTrainer
from particular_model_trainers.losses import SupConLoss

from datasets.uwg_features import get_features

class AppearanceTrainer(AcumenTrainer):

    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model

        self._optimizer = None

        self.criterion = SupConLoss(temperature = self.args.loss_args['temperature'])
        self.appearance_criterion = torch.nn.BCELoss()

    def training_step(self, batch, batch_idx):
        output = self.model(batch['image'])

        appearance_labels = torch.dstack([
            batch[f].reshape((-1, 1)) for f in get_features(self.args)
        ]).squeeze()

        appearance_loss = self.appearance_criterion(output['appearance'], appearance_labels)

        final_loss = appearance_loss

        self.log('train/appearance_loss', appearance_loss.item(), on_step = True)
        self.log('train/loss', final_loss.item(), on_step = True)

        return final_loss
