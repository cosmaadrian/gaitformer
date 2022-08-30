import yaml
from datasets import UWGDense, CASIADataset, FVGDataset, CASIAGenderDataset, FVGGenderDataset, UWGOld, PAIDataset, OUISIRDataset, GREWDataset
from models import GaitFormer, STGCNModel
from particular_model_trainers import WeakMultiTaskTrainer, WildGaitTrainer, SimCLRTrainer, AppearanceTrainer, RecognitionFineTunerTrainer, GenderFineTunerTrainer, GenderTrainer

from evaluators import CASIARecognitionEvaluator, FVGGenderEvaluator, FVGRecognitionEvaluator, CASIAGenderEvaluator, PAIEvaluator, PAIEvaluatorFine

import torch

device = torch.device('cuda')

DATASETS = {
    'uwg-dense': UWGDense,
    'uwg-old': UWGOld,
    'ouisir': OUISIRDataset,
    'grew': GREWDataset,

    'pai': PAIDataset,

    'casia': CASIADataset,
    'fvg': FVGDataset,

    'casia-gender': CASIAGenderDataset,
    'fvg-gender': FVGGenderDataset,
}

EVALUATORS = {
    'casia-recognition': CASIARecognitionEvaluator,
    'casia-gender': CASIAGenderEvaluator,

    'fvg-recognition': FVGRecognitionEvaluator,
    'fvg-gender': FVGGenderEvaluator,

    'pai': PAIEvaluator,
    'pai-fine': PAIEvaluatorFine,
}

TRAINER = {
    'weak-multi-task': WeakMultiTaskTrainer,
    'wildgait': WildGaitTrainer,
    'simclr': SimCLRTrainer,
    'appearance': AppearanceTrainer,
    'gender': GenderTrainer,

    'recognition-fine-tuner': RecognitionFineTunerTrainer,
    'gender-fine-tuner': GenderFineTunerTrainer,
}

MODELS = {
    'stgcn': STGCNModel,
    'gaitformer': GaitFormer,
}

SCHEDULERS = {
    'cyclic': torch.optim.lr_scheduler.CyclicLR
}
