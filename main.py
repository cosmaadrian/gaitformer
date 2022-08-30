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
from utils import load_args

import nomenclature

parser = argparse.ArgumentParser(description='Do stuff.')
parser.add_argument('--config_file', type = str, required = True)
parser.add_argument('--name', type = str, default = 'test')
parser.add_argument('--group', type = str, default = 'default')
parser.add_argument('--notes', type = str, default = '')
parser.add_argument("--mode", type = str, default = 'dryrun')
parser.add_argument('--epochs', type=int, default = None)
parser.add_argument('--batch_size', type=int, default = None)
parser.add_argument('--log_every', type = int, default = 5)
parser.add_argument('--eval_every', type = int, default = None)
parser.add_argument('--accumulation_steps', type = int, default = None)

parser.add_argument('--output_dir', type = str, default = 'dummy')

parser.add_argument('--runs', type = int, default = None)
parser.add_argument('--fraction', type = float, default = None)

parser.add_argument('--env', type = str, default = 'genesis')

parser.add_argument('--trainer', type=str, default = None)
parser.add_argument('--model', type=str, default = None)
parser.add_argument('--dataset', type=str, default = None)

args = parser.parse_args()

args, cfg = load_args(args)
with open('configs/env_config.yaml', 'rt') as f:
    env_cfg = yaml.load(f, Loader = yaml.FullLoader)
args.environment = env_cfg[args.env]

pprint.pprint(args.__dict__)

os.environ['WANDB_MODE'] = args.mode
os.environ['WANDB_NAME'] = args.name
os.environ['WANDB_NOTES'] = args.notes

wandb.init(project = 'gaitformer', group = args.group)

dataset = nomenclature.DATASETS[args.dataset]

wandb.config.update({'dataset_path': dataset.DATASET_PATH})
wandb.config.update(vars(args))
wandb.config.update({'config': cfg})

train_dataloader = nomenclature.DATASETS[args.dataset].train_dataloader(args)
print('::::::::::::::: len data ::::', len(train_dataloader))

architecture = nomenclature.MODELS[args.model](args)
model = nomenclature.TRAINER[args.trainer](args, architecture)

evaluators = [
    nomenclature.EVALUATORS[evaluator_name](args, architecture)
    for evaluator_name in args.evaluators
]

# TODO THIS IS A HACK
# if args.dataset == 'casia':
#     evaluators = [nomenclature.EVALUATORS['casia-recognition'](args, architecture)]
# elif args.dataset == 'fvg':
#     evaluators = [nomenclature.EVALUATORS['fvg-recognition'](args, architecture)]
# else:
#     evaluators = [nomenclature.EVALUATORS[args.dataset](args, architecture)]
monitor_quantity = f'val_acc_{evaluators[0].__class__.__name__}'

#######################################################

wandb_logger = WandbLogger()

### TODO
checkpoint_callback = callbacks.ModelCheckpoint(
    monitor = f'{monitor_quantity}',
    dirpath = f'checkpoints/{args.group}:{args.name}',
    save_weights_only = True,
    direction='up',
    filename=f'epoch={{epoch}}-val_acc={{{monitor_quantity}:.4f}}.ckpt',
    save_to_drive = args.save_to_drive
)

lr_callback = callbacks.MultiLRSchedule(schedulers = [
    (nomenclature.SCHEDULERS[scheduler_args['name']](
        optimizer = model.configure_optimizers(lr = scheduler_args['base_lr']),
        cycle_momentum = False,
        base_lr = scheduler_args['base_lr'],
        mode = scheduler_args['mode'],
        step_size_up = len(train_dataloader) * scheduler_args['step_size_up'], # per epoch
        step_size_down = len(train_dataloader) * scheduler_args['step_size_down'], # per epoch
        max_lr = scheduler_args['max_lr']
    ), scheduler_args['start_epoch'], scheduler_args['end_epoch'])
    for scheduler_args in args.lr_scheduler
])

lr_logger = callbacks.LambdaCallback(
    on_batch_end = lambda: wandb_logger.log('lr', lr_callback.get_last_lr()[0])
)

trainer = NotALightningTrainer(
    args = args,
    callbacks = [
        checkpoint_callback,
        lr_callback,
        lr_logger
    ],
    logger=wandb_logger,
)

torch.backends.cudnn.benchmark = True
trainer.fit(
    model,
    train_dataloader,
    evaluators = evaluators
)
