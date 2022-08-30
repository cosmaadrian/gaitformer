#!/bin/bash
set -e

cd ..

### CONTRASTIVE
python evaluate.py --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/recognition.yaml --model gaitformer --mode run --group experiments-real --name contrastive-sm --env genesis --output_dir direct-transfer
python evaluate.py --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/recognition.yaml --model gaitformer --mode run --group experiments-real --name contrastive --env genesis --output_dir direct-transfer
python evaluate.py --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/recognition.yaml --model gaitformer --mode run --group experiments-real --name contrastive-xl --env genesis --output_dir direct-transfer

################################################################################################################
############### APPEARANCE ONLY XL #######################
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai-gender.yaml --config_file configs/model_sizes/xl.yaml --model gaitformer --mode run --group final-table --name app-all-xl --output_dir direct-transfer
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai-gender.yaml --config_file configs/experiments/appearance-demographic-xl.yaml --model gaitformer --mode run --group final-table --name app-dem-xl --output_dir direct-transfer
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai.yaml --config_file configs/experiments/appearance-clothing-xl.yaml --model gaitformer --mode run --group final-table --name app-cl-xl --output_dir direct-transfer

############### APPEARANCE ONLY MD #######################
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai-gender.yaml --config_file configs/model_sizes/md.yaml --model gaitformer --mode run --group final-table --name app-all-md --output_dir direct-transfer
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai-gender.yaml --config_file configs/experiments/appearance-demographic-md.yaml --model gaitformer --mode run --group final-table --name app-dem-md --output_dir direct-transfer
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai.yaml --config_file configs/experiments/appearance-clothing-md.yaml --model gaitformer --mode run --group final-table --name app-cl-md --output_dir direct-transfer

############### APPEARANCE ONLY SM #######################
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai-gender.yaml --config_file configs/model_sizes/sm.yaml --model gaitformer --mode run --group final-table --name app-all-sm --output_dir direct-transfer
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai-gender.yaml --config_file configs/experiments/appearance-demographic-sm.yaml --model gaitformer --mode run --group final-table --name app-dem-sm --output_dir direct-transfer
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai.yaml --config_file configs/experiments/appearance-clothing-sm.yaml --model gaitformer --mode run --group final-table --name app-cl-sm --output_dir direct-transfer

################################################################################################################
############### Multi-Task XL #############################
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai-gender.yaml --config_file configs/model_sizes/xl.yaml --model gaitformer --mode run --group final-table --name mt-all-xl --output_dir direct-transfer
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai-gender.yaml --config_file configs/experiments/appearance-demographic-xl.yaml --model gaitformer --mode run --group final-table --name mt-dem-xl --output_dir direct-transfer
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai.yaml --config_file configs/experiments/appearance-clothing-xl.yaml --model gaitformer --mode run --group final-table --name mt-cl-xl --output_dir direct-transfer

############### Multi-Task MD #############################
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai-gender.yaml --config_file configs/model_sizes/md.yaml --model gaitformer --mode run --group final-table --name mt-all-md --output_dir direct-transfer
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai-gender.yaml --config_file configs/experiments/appearance-demographic-md.yaml --model gaitformer --mode run --group final-table --name mt-dem-md --output_dir direct-transfer
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai.yaml --config_file configs/experiments/appearance-clothing-md.yaml --model gaitformer --mode run --group final-table --name mt-cl-md --output_dir direct-transfer

############### Multi-Task SM #############################
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai-gender.yaml --config_file configs/model_sizes/sm.yaml --model gaitformer --mode run --group final-table --name mt-all-sm --output_dir direct-transfer
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai-gender.yaml --config_file configs/experiments/appearance-demographic-sm.yaml --model gaitformer --mode run --group final-table --name mt-dem-sm --output_dir direct-transfer
python evaluate.py --env genesis --eval_config configs/evaluation/recognition-pai.yaml --config_file configs/experiments/appearance-clothing-sm.yaml --model gaitformer --mode run --group final-table --name mt-cl-sm --output_dir direct-transfer
