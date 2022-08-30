#/bin/bash
set -e

cd ..


### CASIA
# XL
python evaluate.py --name scratch-casia-1-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-casia-2-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-casia-3-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-casia-5-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-casia-7-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-casia-10-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch

## MD
python evaluate.py --name scratch-casia-1-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-casia-2-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-casia-3-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-casia-5-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-casia-7-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-casia-10-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch

# SM
python evaluate.py --name scratch-casia-1-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-casia-2-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-casia-3-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-casia-5-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-casia-7-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-casia-10-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/casia-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch

#### FVG
## XL
python evaluate.py --name scratch-fvg-1-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-fvg-2-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-fvg-3-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-fvg-5-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-fvg-7-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-fvg-10-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch

## MD
python evaluate.py --name scratch-fvg-1-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-fvg-2-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-fvg-3-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-fvg-5-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-fvg-7-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-fvg-10-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch

## SM
python evaluate.py --name scratch-fvg-1-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-fvg-2-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-fvg-3-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-fvg-5-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-fvg-7-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
python evaluate.py --name scratch-fvg-10-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch
