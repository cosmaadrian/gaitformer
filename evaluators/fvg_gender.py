import argparse
import torch
import wandb
import yaml
import numpy as np
import os

import pandas as pd
from tqdm import tqdm
import json
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import DataLoader

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from datasets.uwg_features import get_features
from datasets import FVGDataset
from evaluators import BaseEvaluator
from datasets.transforms import DeterministicCrop, ToTensor

import nomenclature

class FVGGenderEvaluator(BaseEvaluator):
    def __init__(self, args, model):
        super().__init__(args, model)

        self.dataset = nomenclature.DATASETS['fvg-gender']
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
        for kind, get_split in [
                    ('WS', self.walking_speed_split),
                    ('CL', self.changing_clothes_split),
                    ('BG', self.carrying_bag_split),
                    ('ALL', self.all_split)
                ]:

            data = get_split()
            with torch.no_grad():
                y_pred = self._predict(data)

            data['predictions'] = y_pred
            data['predictions_rounded'] = np.round(y_pred)


            p = precision_score(data['gender'], data['predictions_rounded'], average = 'weighted')
            r = recall_score(data['gender'], data['predictions_rounded'], average = 'weighted')
            f1 = f1_score(data['gender'], data['predictions_rounded'], average = 'weighted')
            a = accuracy_score(data['gender'], data['predictions_rounded'])

            roc_auc = roc_auc_score(data['gender'], data['predictions'], average = 'weighted')
            ap = average_precision_score(data['gender'], data['predictions'], average = 'weighted')

            results.append({
                'kind': kind,
                'precision': p,
                'recall': r,
                'f1': f1,
                'accuracy': a,
                'roc_auc': roc_auc,
                'ap': ap
            })

        wandb.log({f'FVG-Gender': wandb.Table(dataframe = pd.DataFrame(results))})

        if save:
            with open(f'results/{self.args.output_dir}/{self.args.group}:{self.args.name}_fvg-gender.csv', 'wt') as f:
                json.dump(results, f)

        return results

    def _predict(self, annotations):
        gender_index = get_features(self.args).index('Female')
        return torch.cat([
            self.model(data['image'].to(nomenclature.device))['appearance'][:, gender_index]
            for data in tqdm(DataLoader(
                FVGDataset(
                    args = self.args,
                    annotations = annotations,
                    image_transforms = transforms.Compose([
                        DeterministicCrop(period_length = self.args.period_length),
                        ToTensor()
                    ]), kind = 'val'),
                batch_size = 128,
                shuffle = False
            ))
        ]).detach().cpu().numpy()

    def walking_speed_split(self):
        annotations = self.val_dataset.annotations
        return pd.concat([
            (annotations[
                (annotations['session_id'] == 0) &
                (annotations['run_id'].isin([3, 4, 5, 6, 7, 8]))]
            ), (annotations[
                (annotations['session_id'] == 1) &
                (annotations['run_id'].isin([3, 4, 5]))]
            )
        ]).reset_index(drop = True)

    def carrying_bag_split(self):
        annotations = self.val_dataset.annotations

        return annotations[
            (annotations['session_id'] == 0) &
            (annotations['run_id'].isin([9, 10, 11]))
        ].reset_index(drop = True)

    def changing_clothes_split(self):
        annotations = self.val_dataset.annotations

        return annotations[
            (annotations['session_id'] == 1) &
            (annotations['run_id'].isin([6, 7, 8]))
        ].reset_index(drop = True)

    def cluttered_background_split(self):
        annotations = self.val_dataset.annotations

        return annotations[
            (annotations['session_id'] == 1) &
            (annotations['run_id'].isin([9, 10, 11]))
        ].reset_index(drop = True)

    def all_split(self):
        annotations = self.val_dataset.annotations

        return pd.concat([
            (annotations[
                (annotations['session_id'] == 0) &
                (annotations['run_id'].isin([3, 4, 5, 6, 7, 8]))]
            ), (annotations[
                (annotations['session_id'] == 1) &
                (annotations['run_id'].isin([3, 4, 5]))]
            ), (annotations[
                (annotations['session_id'] == 2) &
                (annotations['run_id'].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))]
            )
        ]).reset_index(drop = True)
