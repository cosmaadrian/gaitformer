#!/bin/bash

set -e
cd ..

################################################################################################################
############### APPEARANCE ONLY XL #######################
python evaluate.py --env exodus --eval_config configs/evaluation/pai-fine.yaml --config_file configs/model_sizes/xl.yaml --model gaitformer --mode dryrun --group final-table --name app-all-xl --output_dir pai-fine

############### APPEARANCE ONLY MD #######################
python evaluate.py --env exodus --eval_config configs/evaluation/pai-fine.yaml --config_file configs/model_sizes/md.yaml --model gaitformer --mode dryrun --group final-table --name app-all-md --output_dir pai-fine

############### APPEARANCE ONLY SM #######################
python evaluate.py --env exodus --eval_config configs/evaluation/pai-fine.yaml --config_file configs/model_sizes/sm.yaml --model gaitformer --mode dryrun --group final-table --name app-all-sm --output_dir pai-fine

################################################################################################################
############### Multi-Task XL #############################
python evaluate.py --env exodus --eval_config configs/evaluation/pai-fine.yaml --config_file configs/model_sizes/xl.yaml --model gaitformer --mode dryrun --group final-table --name mt-all-xl --output_dir pai-fine

############### Multi-Task MD #############################
python evaluate.py --env exodus --eval_config configs/evaluation/pai-fine.yaml --config_file configs/model_sizes/md.yaml --model gaitformer --mode dryrun --group final-table --name mt-all-md --output_dir pai-fine

############### Multi-Task SM #############################
python evaluate.py --env exodus --eval_config configs/evaluation/pai-fine.yaml --config_file configs/model_sizes/sm.yaml --model gaitformer --mode dryrun --group final-table --name mt-all-sm --output_dir pai-fine
