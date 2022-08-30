import argparse
import torch
import wandb
import json
import yaml
import numpy as np
import os

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import DataLoader

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from datasets.uwg_features import get_features
from evaluators import BaseEvaluator

import nomenclature

class CASIAGenderEvaluator(BaseEvaluator):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.dataset = nomenclature.DATASETS['casia-gender']
        self.val_dataset = self.dataset(args = args, kind = 'val')
        self.val_dataloader = DataLoader(self.val_dataset, batch_size = args.batch_size, num_workers = 4, pin_memory = False, shuffle = False)

    def trainer_evaluate(self, step):
        print("Running Evaluation.")
        results = self.evaluate(save = False)
        return results[-1]['f1']

    def evaluate(self, save = True):
        y_pred = []
        annotations = self.val_dataset.annotations

        gender_index = get_features(self.args).index('Female')

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_dataloader, total = len(self.val_dataloader))):
                output = self.model(batch['image'].to(nomenclature.device))
                gender_batch = output['appearance'][:, gender_index]
                gender_batch = gender_batch.detach().cpu().numpy()
                y_pred.extend(gender_batch.tolist())

        annotations['predictions'] = y_pred
        annotations['predictions_rounded'] = np.round(y_pred)
        annotations = annotations.drop('file_name', axis = 1)

        results = []

        results_df = pd.DataFrame(columns = ['angle', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'ap'])
        for i, (name, group) in enumerate(annotations.groupby('camera_id')):
            p = precision_score(group['gender'], group['predictions_rounded'], average = 'weighted')
            r = recall_score(group['gender'], group['predictions_rounded'], average = 'weighted')
            f1 = f1_score(group['gender'], group['predictions_rounded'], average = 'weighted')
            a = accuracy_score(group['gender'], group['predictions_rounded'])

            roc_auc = roc_auc_score(group['gender'], group['predictions'], average = 'weighted')
            ap = average_precision_score(group['gender'], group['predictions'], average = 'weighted')

            results.append({
                'angle': name,
                'precision': p,
                'recall': r,
                'f1': f1,
                'accuracy': a,
                'roc_auc': roc_auc,
                'ap': ap
            })

        p = precision_score(annotations['gender'], annotations['predictions_rounded'], average = 'weighted')
        r = recall_score(annotations['gender'], annotations['predictions_rounded'], average = 'weighted')
        f1 = f1_score(annotations['gender'], annotations['predictions_rounded'], average = 'weighted')
        a = accuracy_score(annotations['gender'], annotations['predictions_rounded'])

        roc_auc = roc_auc_score(annotations['gender'], annotations['predictions'], average = 'weighted')
        ap = average_precision_score(annotations['gender'], annotations['predictions'], average = 'weighted')

        results.append({
            'angle': None,
            'precision': p,
            'recall': r,
            'f1': f1,
            'accuracy': a,
            'roc_auc': roc_auc,
            'ap': ap
        })

        wandb.log({f'CASIA-Gender': wandb.Table(dataframe = pd.DataFrame(results))})

        if save:
            with open(f'results/{self.args.output_dir}/{self.args.group}:{self.args.name}_casia-gender.csv', 'wt') as f:
                json.dump(results, f)

        return results