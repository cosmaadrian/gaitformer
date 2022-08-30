#!/bin/bash
set -e

# ballpark comparison with only 20 epochs between models and types of training

cd ..
#######################################################################
###### model gaitformer-inject
#######################################################################
python main.py --config_file configs/base_config.yaml --trainer weak-multi-task --dataset uwg-dense --model gaitformer-inject --mode run --group experiments --name gaitformer-inject --epochs 200

#######################################################################
###### model gaitformer
#######################################################################
python main.py --config_file configs/base_config.yaml --trainer appearance --dataset uwg-dense --model gaitformer --mode run --group comparison-gaitformer --name appearance-only --epochs 20
python main.py --config_file configs/base_config.yaml --trainer wildgait --dataset uwg-dense --model gaitformer --mode run --group comparison-gaitformer --name wildgait --epochs 20
python main.py --config_file configs/base_config.yaml --trainer simclr --dataset uwg-dense --model gaitformer --mode run --group comparison-gaitformer --name simclr --epochs 20
python main.py --config_file configs/base_config.yaml --trainer weak-multi-task --dataset uwg-dense --model gaitformer --mode run --group comparison-gaitformer --name multi-task --epochs 20

#######################################################################
###### model st-gcn
#######################################################################
python main.py --config_file configs/base_config.yaml --trainer appearance --dataset uwg-dense --model stgcn --mode run --group comparison-gcn --name appearance-only --epochs 20
python main.py --config_file configs/base_config.yaml --trainer wildgait --dataset uwg-dense --model stgcn --mode run --group comparison-gcn --name wildgait --epochs 20
python main.py --config_file configs/base_config.yaml --trainer simclr --dataset uwg-dense --model stgcn --mode run --group comparison-gcn --name simclr --epochs 20
python main.py --config_file configs/base_config.yaml --trainer weak-multi-task --dataset uwg-dense --model stgcn --mode run --group comparison-gcn --name multi-task --epochs 20


#######################################################################
###### model gcn-transformer
#######################################################################
python main.py --config_file configs/base_config.yaml --trainer appearance --dataset uwg-dense --model gcn-transformer --mode run --group comparison-gcn-transformer --name appearance-only --epochs 20
python main.py --config_file configs/base_config.yaml --trainer wildgait --dataset uwg-dense --model gcn-transformer --mode run --group comparison-gcn-transformer --name wildgait --epochs 20
python main.py --config_file configs/base_config.yaml --trainer simclr --dataset uwg-dense --model gcn-transformer --mode run --group comparison-gcn-transformer --name simclr --epochs 20
python main.py --config_file configs/base_config.yaml --trainer weak-multi-task --dataset uwg-dense --model gcn-transformer --mode run --group comparison-gcn-transformer --name multi-task --epochs 20
