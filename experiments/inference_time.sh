#!/bin/bash
set -e
cd ..

python evaluate_inference_time.py --config_file configs/model_sizes/sm.yaml --model gaitformer --name "GaitFormerSM"
python evaluate_inference_time.py --config_file configs/model_sizes/md.yaml --model gaitformer --name "GaitFormerMD"
python evaluate_inference_time.py --config_file configs/model_sizes/xl.yaml --model gaitformer --name "GaitFormerXL"

python evaluate_inference_time.py --config_file configs/stgcn-pretrain.yaml --model stgcn --name "ST-GCN"
