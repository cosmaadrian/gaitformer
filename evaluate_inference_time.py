import time
import json
import numpy as np
import torch
import argparse
import yaml
import sys

import nomenclature
from utils import load_args

parser = argparse.ArgumentParser(description='Do stuff.')
parser.add_argument('--config_file', type = str, required = True)
parser.add_argument('--model', type = str, required = True)
parser.add_argument('--name', type = str, required = True)
args = parser.parse_args()
args, cfg = load_args(args)

print(args)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

output = {
    'period_length': [],
    'time-mean': [],
    'time-std': []
}

for period_length in [12, 24, 36, 48, 60, 72, 84, 96]:
    mock_input = torch.rand((512, 3, period_length, 18, 1)).cuda()
    times = []

    args.period_length = period_length
    model = nomenclature.MODELS[args.model](args).cuda()
    model.eval()
    model.train(False)
    num_params = count_parameters(model)
    print(args.name, 'params:', num_params)

    for _ in range(100):
        with torch.no_grad():
            start_time = time.time()
            model(mock_input)
            end_time = time.time()
            times.append(end_time - start_time)

    output['period_length'].append(period_length)
    output['time-mean'].append(float(np.mean(times)))
    output['time-std'].append(float(np.std(times)))

with open(f'experiments/inference/{args.name}.json', 'wt') as f:
    json.dump(output, f)
