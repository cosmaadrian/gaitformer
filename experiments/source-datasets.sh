#!/bin/bash
set -e
cd ..

###############################################################

# train GaitFormer on GREW
python main.py --config_file configs/base_config.yaml  --dataset grew --model gaitformer --trainer wildgait  --mode run --group journal-exp-2 --name gaitformer-grew --epochs 200 --env exodus
python main.py --config_file configs/base_config.yaml  --dataset grew --model gaitformer --trainer wildgait  --mode run --group journal-exp-2 --name gaitformer-grew-vanilla --epochs 400 --env exodus

# train GaitFormer on UWG
python main.py --config_file configs/base_config.yaml  --dataset uwg-old --model gaitformer --trainer wildgait  --mode run --group journal-exp-2 --name gaitformer-uwg --epochs 200 --env exodus

# train GaitFormer on OU-ISIR
python main.py --config_file configs/base_config.yaml  --dataset ouisir --model gaitformer --trainer wildgait  --mode run --group journal-exp-2 --name gaitformer-ouisir --epochs 200 --env exodus

################################################################
################################################################
################################################################
################################################################
#########################33


# train stgcn on GREW
python main.py --config_file configs/stgcn-pretrain.yaml  --dataset grew --model stgcn --trainer wildgait  --mode run --group journal-exp-2 --name stgcn-grew --epochs 200 --env exodus
# # train stgcn on DenseGait Multi-Task
python main.py --config_file configs/stgcn-pretrain.yaml  --dataset uwg-dense --model stgcn --trainer weak-multi-task --mode run --group journal-exp-2 --name stgcn-densegait-mt --epochs 200 --env exodus

# train stgcn on DenseGait
python main.py --config_file configs/stgcn-pretrain.yaml  --dataset uwg-dense --model stgcn --trainer wildgait --mode run --group journal-exp-2 --name stgcn-densegait --epochs 200 --env exodus

# train stgcn on OUISIR
python main.py --config_file configs/stgcn-pretrain.yaml  --dataset ouisir --model stgcn --trainer wildgait --mode run --group journal-exp-2 --name stgcn-ouisir --epochs 200 --env exodus
###########################################################################
