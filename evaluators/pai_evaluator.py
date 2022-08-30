import argparse
import torch
import wandb
import yaml
import numpy as np
import os
import json

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from datasets.uwg_features import get_features

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import nomenclature
from evaluators.utils import *
from evaluators import BaseEvaluator

class PAIEvaluator(BaseEvaluator):
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

        results_df = pd.DataFrame(columns = ['au_roc', 'accuracy', 'precision', 'recall', 'f1'])

        y_pred_rounded = np.round(y_pred)
        y_true_rounded = np.round(y_true)

        results = {
            'mse': np.mean((y_true - y_pred) ** 2),
            'log_loss': log_loss(y_true, y_pred),
            'hamming_loss': hamming_loss(y_true_rounded, y_pred_rounded),
            'emr': emr(y_true_rounded, y_pred_rounded),
            'example_acc': example_based_accuracy(y_true_rounded, y_pred_rounded),
            'example_prec': example_based_precision(y_true_rounded, y_pred_rounded),
            'label_macro_acc': label_based_macro_accuracy(y_true_rounded, y_pred_rounded),
            'label_macro_prec': label_based_macro_precision(y_true_rounded, y_pred_rounded),
            'label_macro_recall': label_based_macro_recall(y_true_rounded, y_pred_rounded),
            'label_micro_acc': label_based_micro_accuracy(y_true_rounded, y_pred_rounded),
            'label_micro_prec': label_based_micro_precision(y_true_rounded, y_pred_rounded),
            'label_micro_recall': label_based_micro_recall(y_true_rounded, y_pred_rounded),
            'alpha_eval': alpha_evaluation_score(y_true_rounded, y_pred_rounded),
        }

        wandb.log({f'PAI': wandb.Table(dataframe = pd.DataFrame(results, index = [0]))})

        if save:
            with open(f'results/{self.args.output_dir}/{self.args.group}:{self.args.name}_pai-fidelity.csv', 'wt') as f:
                json.dump(results, f)

        return results
