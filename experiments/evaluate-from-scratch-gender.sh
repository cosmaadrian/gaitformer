#/bin/bash
set -e

cd ..


### CASIA
# XL
python evaluate.py --name scratch-casia-gender-1-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-casia-gender-2-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-casia-gender-3-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-casia-gender-5-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-casia-gender-7-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-casia-gender-10-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender

## MD
python evaluate.py --name scratch-casia-gender-1-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-casia-gender-2-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-casia-gender-3-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-casia-gender-5-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-casia-gender-7-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-casia-gender-10-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender

# SM
python evaluate.py --name scratch-casia-gender-1-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-casia-gender-2-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-casia-gender-3-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-casia-gender-5-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-casia-gender-7-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-casia-gender-10-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender

#### FVG
## XL
python evaluate.py --name scratch-fvg-gender-1-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-fvg-gender-2-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-fvg-gender-3-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-fvg-gender-5-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-fvg-gender-7-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-fvg-gender-10-xl --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender

## MD
python evaluate.py --name scratch-fvg-gender-1-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-fvg-gender-2-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-fvg-gender-3-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-fvg-gender-5-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-fvg-gender-7-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-fvg-gender-10-md --config_file configs/model_sizes/md.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender

## SM
python evaluate.py --name scratch-fvg-gender-1-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-fvg-gender-2-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-fvg-gender-3-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-fvg-gender-5-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-fvg-gender-7-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
python evaluate.py --name scratch-fvg-gender-10-sm --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml  --model gaitformer --mode run --group from_scratch  --env exodus --output_dir from-scratch-gender
