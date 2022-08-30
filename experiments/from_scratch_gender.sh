#!/bin/bash
set -e
cd ..

######## CASIA
#### XL
python main.py --fraction 0.1 --name scratch-casia-gender-1-xl --config_file configs/model_sizes/xl.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.2 --name scratch-casia-gender-2-xl --config_file configs/model_sizes/xl.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.3 --name scratch-casia-gender-3-xl --config_file configs/model_sizes/xl.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.5 --name scratch-casia-gender-5-xl --config_file configs/model_sizes/xl.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.7 --name scratch-casia-gender-7-xl --config_file configs/model_sizes/xl.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 1 --name scratch-casia-gender-10-xl --config_file configs/model_sizes/xl.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus

#### MD
python main.py --fraction 0.1 --name scratch-casia-gender-1-md --config_file configs/model_sizes/md.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.2 --name scratch-casia-gender-2-md --config_file configs/model_sizes/md.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.3 --name scratch-casia-gender-3-md --config_file configs/model_sizes/md.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.5 --name scratch-casia-gender-5-md --config_file configs/model_sizes/md.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.7 --name scratch-casia-gender-7-md --config_file configs/model_sizes/md.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 1 --name scratch-casia-gender-10-md --config_file configs/model_sizes/md.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus

#### SM
python main.py --fraction 0.1 --name scratch-casia-gender-1-sm --config_file configs/model_sizes/sm.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.2 --name scratch-casia-gender-2-sm --config_file configs/model_sizes/sm.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.3 --name scratch-casia-gender-3-sm --config_file configs/model_sizes/sm.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.5 --name scratch-casia-gender-5-sm --config_file configs/model_sizes/sm.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.7 --name scratch-casia-gender-7-sm --config_file configs/model_sizes/sm.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 1 --name scratch-casia-gender-10-sm --config_file configs/model_sizes/sm.yaml --trainer gender --dataset casia-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus


####### FVG
### XL
python main.py --fraction 0.1 --name scratch-fvg-gender-1-xl --config_file configs/model_sizes/xl.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.2 --name scratch-fvg-gender-2-xl --config_file configs/model_sizes/xl.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.3 --name scratch-fvg-gender-3-xl --config_file configs/model_sizes/xl.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.5 --name scratch-fvg-gender-5-xl --config_file configs/model_sizes/xl.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.7 --name scratch-fvg-gender-7-xl --config_file configs/model_sizes/xl.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 1 --name scratch-fvg-gender-10-xl --config_file configs/model_sizes/xl.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus

#### MD
python main.py --fraction 0.1 --name scratch-fvg-gender-1-md --config_file configs/model_sizes/md.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.2 --name scratch-fvg-gender-2-md --config_file configs/model_sizes/md.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.3 --name scratch-fvg-gender-3-md --config_file configs/model_sizes/md.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.5 --name scratch-fvg-gender-5-md --config_file configs/model_sizes/md.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.7 --name scratch-fvg-gender-7-md --config_file configs/model_sizes/md.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 1 --name scratch-fvg-gender-10-md --config_file configs/model_sizes/md.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus


### SM
python main.py --fraction 0.1 --name scratch-fvg-gender-1-sm --config_file configs/model_sizes/sm.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.2 --name scratch-fvg-gender-2-sm --config_file configs/model_sizes/sm.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.3 --name scratch-fvg-gender-3-sm --config_file configs/model_sizes/sm.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.5 --name scratch-fvg-gender-5-sm --config_file configs/model_sizes/sm.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 0.7 --name scratch-fvg-gender-7-sm --config_file configs/model_sizes/sm.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
python main.py --fraction 1 --name scratch-fvg-gender-10-sm --config_file configs/model_sizes/sm.yaml --trainer gender --dataset fvg-gender --model gaitformer --mode run --group from_scratch  --epochs 200 --env exodus
