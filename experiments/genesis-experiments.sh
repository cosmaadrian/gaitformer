#!/bin/bash
set -e
cd ..

# comparison between training methods, with gaitformer
python main.py --config_file configs/base_config.yaml --trainer wildgait --dataset uwg-dense --model gaitformer --mode run --group experiments-real --name contrastive-sm --epochs 400

############### APPEARANCE ONLY SM #######################
python main.py --env genesis --config_file configs/model_sizes/sm.yaml --trainer appearance --dataset uwg-dense --model gaitformer --mode run --group final-table --name app-all-sm --epochs 400
python main.py --env genesis --config_file configs/experiments/appearance-demographic-sm.yaml --trainer appearance --dataset uwg-dense --model gaitformer --mode run --group final-table --name app-dem-sm --epochs 400
python main.py --env genesis --config_file configs/experiments/appearance-clothing-sm.yaml --trainer appearance --dataset uwg-dense --model gaitformer --mode run --group final-table --name app-cl-sm --epochs 400

############### Multi-Task SM #############################
python main.py --env genesis --config_file configs/model_sizes/sm.yaml --trainer weak-multi-task --dataset uwg-dense --model gaitformer --mode run --group final-table --name mt-all-sm --epochs 400
python main.py --env genesis --config_file configs/experiments/appearance-demographic-sm.yaml --trainer weak-multi-task --dataset uwg-dense --model gaitformer --mode run --group final-table --name mt-dem-sm --epochs 400
python main.py --env genesis --config_file configs/experiments/appearance-clothing-sm.yaml --trainer weak-multi-task --dataset uwg-dense --model gaitformer --mode run --group final-table --name mt-cl-sm --epochs 400

############################################################################################################################################
############################################################################################################################################
############################################################################################################################################

############### APPEARANCE ONLY MD #######################
python main.py --env genesis --config_file configs/model_sizes/md.yaml --trainer appearance --dataset uwg-dense --model gaitformer --mode run --group final-table --name app-all-md --epochs 400
python main.py --env genesis --config_file configs/experiments/appearance-demographic-md.yaml --trainer appearance --dataset uwg-dense --model gaitformer --mode run --group final-table --name app-dem-md --epochs 400
python main.py --env genesis --config_file configs/experiments/appearance-clothing-md.yaml --trainer appearance --dataset uwg-dense --model gaitformer --mode run --group final-table --name app-cl-md --epochs 400

############### Multi-Task MD #############################
python main.py --env genesis --config_file configs/model_sizes/md.yaml --trainer weak-multi-task --dataset uwg-dense --model gaitformer --mode run --group final-table --name mt-all-md --epochs 400
python main.py --env genesis --config_file configs/experiments/appearance-demographic-md.yaml --trainer weak-multi-task --dataset uwg-dense --model gaitformer --mode run --group final-table --name mt-dem-md --epochs 400
python main.py --env genesis --config_file configs/experiments/appearance-clothing-md.yaml --trainer weak-multi-task --dataset uwg-dense --model gaitformer --mode run --group final-table --name mt-cl-md --epochs 400
