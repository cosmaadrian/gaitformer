import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .acumen_trainer import AcumenTrainer
from particular_model_trainers.losses import SupConLoss
from particular_model_trainers import RecognitionFineTunerTrainer

from datasets.uwg_features import get_features

class GenderFineTunerTrainer(RecognitionFineTunerTrainer):

    def __init__(self, args, model):
        super().__init__(args, model)
        self.args = args
        self.model = model

        self._optimizer = None

        self.criterion = torch.nn.BCELoss()

    def training_step(self, batch, batch_idx):
        output = self.model(batch['image'])

        gender_index = get_features(self.args).index('Female')

        prediction = output['appearance'][:, gender_index]
        labels = batch['gender'].squeeze()

        loss = self.criterion(prediction, labels)

        self.log('train/gender_loss', loss.item(), on_step = True)

        return loss
