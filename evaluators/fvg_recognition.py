from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
import pprint
import nomenclature
import wandb

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch

from datasets import FVGDataset
from datasets.transforms import RandomCrop, ToTensor, DeterministicCrop

from .base_evaluator import BaseEvaluator

class FVGRecognitionEvaluator(BaseEvaluator):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.dataloader = FVGDataset.val_dataloader(args)

    def trainer_evaluate(self, step):
        log_dict = {
            'WS': self.evaluate_walking_speed(),
            'CB': self.evaluate_carrying_bag(),
            'CL': self.evaluate_changing_clothes(),
            'CBG': self.evaluate_cluttered_background(),
            'ALL': self.evaluate_all()
        }
        val_acc = np.mean([value for key, value in log_dict.items()])
        print("FVG(avg) Evaluation Accuracy:", val_acc)
        pprint.pprint(log_dict)
        wandb.log(log_dict, step = step)
        return val_acc

    def evaluate(self, save = False):
        log_dict = {
            'WS': self.evaluate_walking_speed(),
            'CB': self.evaluate_carrying_bag(),
            'CL': self.evaluate_changing_clothes(),
            'CBG': self.evaluate_cluttered_background(),
            'ALL': self.evaluate_all()
        }
        df = pd.DataFrame(log_dict, index = [0])
        table = wandb.Table(dataframe = df)
        wandb.log({"fvg_eval" : table})

        if save:
            df.to_csv(f'results/{self.args.output_dir}/{self.args.group}_{self.args.name}_FVG.csv', index = False)

        return df

    def _predict(self, annotations):
        return torch.cat([
            self.model(data['image'].to(nomenclature.device))['representation'] for data in tqdm.tqdm(DataLoader(
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

    def evaluate_protocol(self, gallery_walks, probe_walks):
        gallery_walks = gallery_walks.reset_index(drop = True)
        probe_walks = probe_walks.reset_index(drop = True)

        with torch.no_grad():
            gallery_embeddings = self._predict(gallery_walks)
            probe_embeddings = self._predict(probe_walks)

        knn = KNeighborsClassifier(1, p = 1)
        knn.fit(gallery_embeddings, gallery_walks['track_id'].values)

        return knn.score(probe_embeddings, probe_walks['track_id'].values)

    def evaluate_walking_speed(self):
        test_df = self.dataloader.dataset.annotations
        gallery_walks = pd.concat([
            (test_df[
                (test_df['session_id'] == 0) &
                (test_df['run_id'] == 1)]
            ),(test_df[
                (test_df['session_id'] == 1) &
                (test_df['run_id'] == 1)]
            )
        ])

        probe_walks = pd.concat([
            (test_df[
                (test_df['session_id'] == 0) &
                (test_df['run_id'].isin([3, 4, 5, 6, 7, 8]))]
            ), (test_df[
                (test_df['session_id'] == 1) &
                (test_df['run_id'].isin([3, 4, 5]))]
            )
        ])
        return self.evaluate_protocol(gallery_walks, probe_walks)

    def evaluate_carrying_bag(self):
        test_df = self.dataloader.dataset.annotations
        gallery_walks = test_df[
            (test_df['session_id'] == 0) &
            (test_df['run_id'] == 1)
        ]

        probe_walks = test_df[
            (test_df['session_id'] == 0) &
            (test_df['run_id'].isin([9, 10, 11]))
        ]

        return self.evaluate_protocol(gallery_walks, probe_walks)

    def evaluate_changing_clothes(self):
        test_df = self.dataloader.dataset.annotations
        gallery_walks = test_df[
            (test_df['session_id'] == 1) &
            (test_df['run_id'] == 1)
        ]

        probe_walks = test_df[
            (test_df['session_id'] == 1) &
            (test_df['run_id'].isin([6, 7, 8]))
        ]

        return self.evaluate_protocol(gallery_walks, probe_walks)

    def evaluate_cluttered_background(self):
        test_df = self.dataloader.dataset.annotations
        gallery_walks = test_df[
            (test_df['session_id'] == 1) &
            (test_df['run_id'] == 1)
        ]

        probe_walks = test_df[
            (test_df['session_id'] == 1) &
            (test_df['run_id'].isin([9, 10, 11]))
        ]

        return self.evaluate_protocol(gallery_walks, probe_walks)

    def evaluate_all(self):
        test_df = self.dataloader.dataset.annotations
        gallery_walks = pd.concat([
            (test_df[
                (test_df['session_id'] == 0) &
                (test_df['run_id'] == 1)]
            ),(test_df[
                (test_df['session_id'] == 1) &
                (test_df['run_id'] == 1)]
            )
        ])

        probe_walks = pd.concat([
            (test_df[
                (test_df['session_id'] == 0) &
                (test_df['run_id'].isin([3, 4, 5, 6, 7, 8]))]
            ), (test_df[
                (test_df['session_id'] == 1) &
                (test_df['run_id'].isin([3, 4, 5]))]
            ), (test_df[
                (test_df['session_id'] == 2) &
                (test_df['run_id'].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))]
            )
        ])
        return self.evaluate_protocol(gallery_walks, probe_walks)
