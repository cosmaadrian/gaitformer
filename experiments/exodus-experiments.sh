#!/bin/bash
set -e
cd ..

# comparison between training methods, with gaitformer
python main.py --config_file configs/base_config.yaml --trainer wildgait --dataset uwg-dense --model gaitformer --mode run --group experiments-real --name contrastive-sm --epochs 400

############### APPEARANCE ONLY XL #######################
python main.py --env exodus --config_file configs/model_sizes/xl.yaml --trainer appearance --dataset uwg-dense --model gaitformer --mode run --group final-table --name app-all-xl --epochs 400
python main.py --env exodus --config_file configs/experiments/appearance-demographic-xl.yaml --trainer appearance --dataset uwg-dense --model gaitformer --mode run --group final-table --name app-dem-xl --epochs 400
python main.py --env exodus --config_file configs/experiments/appearance-clothing-xl.yaml --trainer appearance --dataset uwg-dense --model gaitformer --mode run --group final-table --name app-cl-xl --epochs 400
############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

############### Multi-Task XL #############################
python main.py --env exodus --config_file configs/model_sizes/xl.yaml --trainer weak-multi-task --dataset uwg-dense --model gaitformer --mode run --group final-table --name mt-all-xl --epochs 400
python main.py --env exodus --config_file configs/experiments/appearance-demographic-xl.yaml --trainer weak-multi-task --dataset uwg-dense --model gaitformer --mode run --group final-table --name mt-dem-xl  --epochs 400
python main.py --env exodus --config_file configs/experiments/appearance-clothing-xl.yaml --trainer weak-multi-task --dataset uwg-dense --model gaitformer --mode run --group final-table --name mt-cl-xl --epochs 400
