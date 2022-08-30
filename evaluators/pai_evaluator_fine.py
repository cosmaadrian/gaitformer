import argparse
import torch
import wandb
import yaml
import numpy as np
import os
import json

import pandas as pd
import pprint
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from datasets.uwg_features import get_features

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.metrics import roc_auc_score, median_absolute_error, r2_score

import nomenclature
from evaluators.utils import *
from evaluators import BaseEvaluator

class PAIEvaluatorFine(BaseEvaluator):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.dataset = nomenclature.DATASETS['pai']
        self.val_dataloader = self.dataset.val_dataloader(args)

    def evaluate(self, save = True):
        y_pred = []
        annotations = self.val_dataloader.dataset.annotations

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_dataloader, total = len(self.val_dataloader))):
                output = self.model(batch['image'].to(nomenclature.device))
                gender_batch = output['appearance'].detach().cpu().numpy()
                y_pred.append(gender_batch.tolist())

        y_pred = np.vstack(y_pred)
        y_true = annotations[get_features(self.args)].values

        y_pred_rounded = np.round(y_pred)
        y_true_rounded = np.round(y_true)

        results = dict()
        for i, feature in enumerate(get_features(self.args)):
            y_pred_feature = y_pred[:, i].reshape((-1, 1))
            y_true_feature = y_true[:, i].reshape((-1, 1))

            y_pred_rounded_feature = y_pred_rounded[:, i]
            y_true_rounded_feature = y_true_rounded[:, i]

            try:
                roc = roc_auc_score(y_true_rounded_feature, y_pred_feature, average = 'weighted')
            except ValueError as e:
                print(':::::::: roc is None', feature)
                roc = None

            results[feature] = {
                'mse': np.mean((y_true_feature - y_pred_feature) ** 2),
                'median_absolute_error': median_absolute_error(y_true_feature, y_pred_feature),
                'mean_absolute_error': mean_absolute_error(y_true_feature, y_pred_feature),
                'roc_auc': roc,
                'r2_score': r2_score(y_true_feature, y_pred_feature),
                'log_loss': log_loss(y_true_feature, y_pred_feature),
                'precision': precision_score(y_true_rounded_feature, y_pred_rounded_feature, average = 'weighted'),
                'recall': recall_score(y_true_rounded_feature, y_pred_rounded_feature, average = 'weighted'),
                'f1': recall_score(y_true_rounded_feature, y_pred_rounded_feature, average = 'weighted'),
                'ap': average_precision_score(y_true_rounded_feature, y_pred_rounded_feature, average = 'weighted'),
            }

        # wandb.log({f'PAI': wandb.Table(dataframe = pd.DataFrame(results, index = [0]))})

        if save:
            with open(f'results/{self.args.output_dir}/{self.args.group}:{self.args.name}_pai-fine.csv', 'wt') as f:
                json.dump(results, f)

        return results
