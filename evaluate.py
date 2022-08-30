import torch
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import torch
import wandb
import yaml
import os

from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import DataLoader
import pprint

import callbacks
from trainer import NotALightningTrainer
from loggers import WandbLogger

from schedulers import LRFinder, OneCycleLR
from utils import load_args, load_model, extend_config

import nomenclature

parser = argparse.ArgumentParser(description='Do stuff.')
parser.add_argument('--config_file', type = str, required = True)
parser.add_argument('--eval_config', type = str, required = True)
parser.add_argument('--name', type = str, default = 'test')
parser.add_argument('--group', type = str, default = 'default')
parser.add_argument('--notes', type = str, default = '')
parser.add_argument("--mode", type = str, default = 'dryrun')
parser.add_argument("--env", type = str, default = 'genesis')
parser.add_argument("--output_dir", type = str, default = 'test')

parser.add_argument('--model', type=str, default = None)

args = parser.parse_args()
args, cfg = load_args(args)

with open(args.eval_config, 'rt') as f:
    eval_cfg = yaml.load(f, Loader = yaml.FullLoader)

with open('configs/env_config.yaml', 'rt') as f:
    env_cfg = yaml.load(f, Loader = yaml.FullLoader)

args.environment = env_cfg[args.env]

pprint.pprint(args.__dict__)
pprint.pprint(eval_cfg)

os.environ['WANDB_MODE'] = args.mode
os.environ['WANDB_NAME'] = args.name
os.environ['WANDB_NOTES'] = args.notes

wandb.init(project = 'gaitformer', group = args.group)

wandb.config.update(vars(args))
wandb.config.update({'config': cfg})

architecture = nomenclature.MODELS[args.model](args)

state_dict = load_model(args)
state_dict = {
    key.replace('module.', ''): value
    for key, value in state_dict.items()
}

architecture.load_state_dict(state_dict)
architecture.eval()
architecture.train(False)
architecture.to(nomenclature.device)

evaluators = [
    nomenclature.EVALUATORS[evaluator_name](args, architecture)
    for evaluator_name in eval_cfg['evaluators']
]

for evaluator in evaluators:
    results = evaluator.evaluate(save = True)
    print(evaluator.__class__.__name__)
    pprint.pprint(results)
