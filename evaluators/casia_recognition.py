from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tqdm
import wandb
import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import nomenclature
import torch

from datasets import CASIADataset
from datasets.transforms import RandomCrop, ToTensor, DeterministicCrop

from .base_evaluator import BaseEvaluator


class CASIARecognitionEvaluator(BaseEvaluator):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.dataloader = CASIADataset.val_dataloader(args)
        self.__train_embeddings = None

    def trainer_evaluate(self, step):
        print("Running Evaluation.")
        eval_results_A = self._evaluate_single(kind = 'A')
        # eval_results_B = self._evaluate_single(kind = 'B')
        # eval_results_C = self._evaluate_single(kind = 'C')

        val_acc = eval_results_A.mean()['Accuracy']

        log_dict = dict()
        for _, row in eval_results_A.iterrows():
            log_dict[f"Angle_{row['Probe Angle']}"] = row['Accuracy']

        # log_dict['clothing_accuracy'] = eval_results_B.mean()['Accuracy']
        # log_dict['bag_accuracy'] = eval_results_C.mean()['Accuracy']

        # print(":::: CL", log_dict['clothing_accuracy'])
        # print(":::: BG", log_dict['bag_accuracy'])

        wandb.log(log_dict, step = step)

        self._clear_cache()
        return val_acc

    def evaluate(self, save = False):
#         fig, ax = self._visualize()
#         wandb.log({'tsne-viz': fig})
        # plt.show()

        set_results = self._evaluate_set()
        wandb.log({'Angle Results': wandb.Table(dataframe = set_results)})

        if save:
            set_results.to_csv(f'results/{self.args.output_dir}/{self.args.group}:{self.args.name}_CASIA_angles.csv', index = False)

        for i, (name, group) in enumerate(set_results.groupby(by = 'Gallery Angle')):
            fig, ax = plt.subplots()
            print(name, group['Accuracy'].mean())
            ax.set_title('Gallery Angle: ' + str(name))
            ax.set_ylim(0.0, 1.0)
            ax.set_yticks(np.arange(0.0, 1.1, 0.1))
            ax.set_xticks(sorted(set_results['Gallery Angle'].unique()))
            ax.grid(b = True, which = 'major', color = 'b', linestyle = '-')
            group.plot(x = 'Probe Angle', y = 'Accuracy', ax = ax, style = '^-')
            # TODO dosent work on exodus
            # wandb.log({f'angle-eval-{str(name)}': fig})

        fig, ax = plt.subplots()
        for kind in 'ABC':
            results = self._evaluate_single(kind = kind)
            results.plot(x = 'Probe Angle', y = 'Accuracy', kind = 'line', ax = ax, label = kind)
            print(kind, results)
            print("====> MEAN:", results.mean())
            wandb.log({f'Results Exp. {kind}': wandb.Table(dataframe = results)})

            if save:
                results.to_csv(f'results/{self.args.output_dir}/{self.args.group}:{self.args.name}_CASIA_{kind}.csv', index = False)

        ax.set_ylim(0.0, 1.0)
        ax.set_yticks(np.arange(0.0, 1.1, 0.1))
        ax.set_xticks(sorted(results['Probe Angle'].unique()))

        # TODO dosent work on exodus
        # wandb.log({f'full-eval': fig})
        self._clear_cache()

        return set_results

    def _clear_cache(self):
        self.__train_embeddings = None

    def _predict(self, annotations, kind = 'gallery'):
        return torch.cat([
            self.model(data['image'].to(nomenclature.device))['representation'] for data in tqdm.tqdm(DataLoader(
                CASIADataset(
                    args = self.args,
                    annotations = annotations,
                    image_transforms = transforms.Compose([
                        DeterministicCrop(period_length = self.args.period_length),
                        ToTensor()
                    ]), kind = 'val'),
                batch_size = 256,
                shuffle = False
            ))
        ]).detach().cpu().numpy()

    def _evaluate_set(self, kind = 'A'):
        test_df = self.dataloader.dataset.annotations
        gallery_walks = test_df[test_df['type'] == 2]
        knn_train = gallery_walks.where(gallery_walks['run_id'] <= 3).dropna().sort_values(by = 'file_name')

        if kind == 'A':
            knn_test = gallery_walks.where(gallery_walks['run_id'] > 3).dropna().sort_values(by = 'file_name')
        elif kind == 'B':
            knn_test = test_df[test_df['type'] == 1].sort_values(by = 'file_name')
        elif kind == 'C':
            knn_test = test_df[test_df['type'] == 0].sort_values(by = 'file_name')
        else:
            raise Exception(f'{kind} is not a valid evaluation protocol.')

        knn_train = knn_train.reset_index(drop = True)
        knn_test = knn_test.reset_index(drop = True)

        with torch.no_grad():
            train_embeddings = self.__train_embeddings
            if train_embeddings is None:
                train_embeddings = self._predict(knn_train, kind = 'gallery')
                self.__train_embeddings = train_embeddings

            test_embeddings = self._predict(knn_test, kind = 'probe')

        results = {
            'Gallery Angle': [],
            'Probe Angle': [],
            'Accuracy': [],
        }

        for camera_id_gallery in sorted(knn_train['camera_id'].unique()):
            for camera_id_probe in sorted(knn_test['camera_id'].unique()):
                knn_gallery = knn_train[knn_train['camera_id'] == camera_id_gallery]
                knn_probe = knn_test[knn_test['camera_id'] == camera_id_probe]

                gallery_embeddings = train_embeddings[knn_gallery.index]
                probe_embeddings = test_embeddings[knn_probe.index]

                y_train = knn_gallery['track_id'].values
                y_test = knn_probe['track_id'].values

                knn = KNeighborsClassifier(1, p = 1)
                knn.fit(gallery_embeddings, y_train)

                y_pred = knn.predict(probe_embeddings)

                accuracy = (y_pred == y_test).mean()

                results['Gallery Angle'].append(camera_id_gallery)
                results['Probe Angle'].append(camera_id_probe)
                results['Accuracy'].append(accuracy)

        results = pd.DataFrame(results)
        return results

    def _visualize(self):
        test_df = self.dataloader.dataset.annotations
        gallery_walks = test_df[test_df['type'] == 2]
        knn_train = gallery_walks.where(gallery_walks['run_id'] <= 3).dropna().sort_values(by = 'file_name')
        knn_train = knn_train.reset_index(drop = True)

        knn_test = gallery_walks.where(gallery_walks['run_id'] > 3).dropna().sort_values(by = 'file_name')
        knn_test = knn_test.reset_index(drop = True)

        with torch.no_grad():
            train_embeddings = self.__train_embeddings
            if train_embeddings is None:
                train_embeddings = self._predict(knn_train, kind = 'gallery')
                self.__train_embeddings = train_embeddings

            test_embeddings = self._predict(knn_test, kind = 'probe')

        tsne = TSNE(2, verbose = 2, early_exaggeration = 32)
        _encoded = tsne.fit_transform(np.vstack((train_embeddings, test_embeddings)))
        encoded = _encoded[:len(train_embeddings)]
        encoded_test = _encoded[len(train_embeddings):]

        fig, ax = plt.subplots(2, 2)
        ax[0, 0].set_title('CameraID')
        ax[0, 0].scatter(x = encoded[:, 0], y = encoded[:, 1], c = knn_train['camera_id'].astype('category').cat.codes.values.ravel())
        ax[1, 0].scatter(x = encoded_test[:, 0], y = encoded_test[:, 1], c = knn_test['camera_id'].astype('category').cat.codes.values.ravel(), marker = 'x')

        ax[0, 1].set_title('TrackID')
        ax[0, 1].scatter(x = encoded[:, 0], y = encoded[:, 1], c = knn_train['track_id'].astype('category').cat.codes.values.ravel())
        ax[1, 1].scatter(x = encoded_test[:, 0], y = encoded_test[:, 1], c = knn_test['track_id'].astype('category').cat.codes.values.ravel(), marker ='x')
        return fig, ax

    def _evaluate_single(self,  kind = 'A', same_angle = False):
        test_df = self.dataloader.dataset.annotations
        gallery_walks = test_df[test_df['type'] == 2]
        knn_train = gallery_walks.where(gallery_walks['run_id'] <= 3).dropna().sort_values(by = 'file_name')

        if kind == 'A':
            knn_test = gallery_walks.where(gallery_walks['run_id'] > 3).dropna().sort_values(by = 'file_name')
        elif kind == 'B':
            knn_test = test_df[test_df['type'] == 1].sort_values(by = 'file_name')
        elif kind == 'C':
            knn_test = test_df[test_df['type'] == 0].sort_values(by = 'file_name')

        knn_train = knn_train.reset_index(drop = True)
        knn_test = knn_test.reset_index(drop = True)

        with torch.no_grad():
            train_embeddings = self.__train_embeddings
            if train_embeddings is None:
                train_embeddings = self._predict(knn_train, kind = 'gallery')
                self.__train_embeddings = train_embeddings

            test_embeddings = self._predict(knn_test, kind = 'probe')

        results = {
            'Probe Angle': [],
            'Accuracy': [],
        }

        knn_gallery = knn_train

        for camera_id_probe in sorted(knn_test['camera_id'].unique()):
            if not same_angle:
                knn_gallery = knn_train[knn_train['camera_id'] != camera_id_probe]

            gallery_embeddings = train_embeddings[knn_gallery.index]

            knn = KNeighborsClassifier(1, p = 1)
            knn.fit(gallery_embeddings, knn_gallery['track_id'].values)

            knn_probe = knn_test[knn_test['camera_id'] == camera_id_probe]

            probe_embeddings = test_embeddings[knn_probe.index]
            y_pred = knn.predict(probe_embeddings)

            accuracy = (y_pred == knn_probe['track_id'].values).mean()

            results['Probe Angle'].append(camera_id_probe)
            results['Accuracy'].append(accuracy)

        results = pd.DataFrame(results)
        return results
