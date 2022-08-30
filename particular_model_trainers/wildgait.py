import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .acumen_trainer import AcumenTrainer
from particular_model_trainers.losses import SupConLoss

class WildGaitTrainer(AcumenTrainer):

    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.model = model

        self._optimizer = None

        self.criterion = SupConLoss(temperature = self.args.loss_args['temperature'])

    def training_step(self, batch, batch_idx):
        output = self.model(batch['image'])

        features = output['projection'].view(-1, self.args.num_views, output['projection'].shape[-1])
        labels = batch['track_id'].squeeze()[::self.args.num_views]

        contrastive_loss = self.criterion(features, labels.squeeze())
        final_loss = contrastive_loss

        self.log('train/recognition_loss', contrastive_loss.item(), on_step = True)
        self.log('train/loss', final_loss.item(), on_step = True)

        return final_loss
