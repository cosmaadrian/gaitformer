import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .acumen_trainer import AcumenTrainer
from particular_model_trainers.losses import SupConLoss

from .losses import SimCLR

class SimCLRTrainer(AcumenTrainer):

    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model

        self._optimizer = None

        self.criterion = SimCLR(args)
        self.appearance_criterion = torch.nn.BCELoss()

    def training_step(self, batch, batch_idx):
        output = self.model(batch['image'])
        features = output['projection']

        simclr_loss = self.criterion(features)
        final_loss = simclr_loss

        self.log('train/recognition_loss', simclr_loss.item(), on_step = True)
        self.log('train/loss', final_loss.item(), on_step = True)

        return final_loss
