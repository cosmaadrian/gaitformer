#!/bin/bash
set -e
cd ..

# ######## CASIA
# #### XL
# python main.py --runs 1 --name scratch-casia-1-xl --config_file configs/model_sizes/xl.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --runs 2 --name scratch-casia-2-xl --config_file configs/model_sizes/xl.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --runs 3 --name scratch-casia-3-xl --config_file configs/model_sizes/xl.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --runs 5 --name scratch-casia-5-xl --config_file configs/model_sizes/xl.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --runs 7 --name scratch-casia-7-xl --config_file configs/model_sizes/xl.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --runs 10 --name scratch-casia-10-xl --config_file configs/model_sizes/xl.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus

# #### MD
# python main.py --runs 1 --name scratch-casia-1-md --config_file configs/model_sizes/md.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --runs 2 --name scratch-casia-2-md --config_file configs/model_sizes/md.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --runs 3 --name scratch-casia-3-md --config_file configs/model_sizes/md.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --runs 5 --name scratch-casia-5-md --config_file configs/model_sizes/md.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --runs 7 --name scratch-casia-7-md --config_file configs/model_sizes/md.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --runs 10 --name scratch-casia-10-md --config_file configs/model_sizes/md.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus

# #### SM
# python main.py --runs 1 --name scratch-casia-1-sm --config_file configs/model_sizes/sm.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --runs 2 --name scratch-casia-2-sm --config_file configs/model_sizes/sm.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --runs 3 --name scratch-casia-3-sm --config_file configs/model_sizes/sm.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --runs 5 --name scratch-casia-5-sm --config_file configs/model_sizes/sm.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --runs 7 --name scratch-casia-7-sm --config_file configs/model_sizes/sm.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --runs 10 --name scratch-casia-10-sm --config_file configs/model_sizes/sm.yaml --trainer wildgait --dataset casia --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus


# ####### FVG
# ### XL
# python main.py --fraction 0.1 --name scratch-fvg-1-xl --config_file configs/model_sizes/xl.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --fraction 0.2 --name scratch-fvg-2-xl --config_file configs/model_sizes/xl.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --fraction 0.3 --name scratch-fvg-3-xl --config_file configs/model_sizes/xl.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --fraction 0.5 --name scratch-fvg-5-xl --config_file configs/model_sizes/xl.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --fraction 0.7 --name scratch-fvg-7-xl --config_file configs/model_sizes/xl.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --fraction 1 --name scratch-fvg-10-xl --config_file configs/model_sizes/xl.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus

# #### MD
# python main.py --fraction 0.1 --name scratch-fvg-1-md --config_file configs/model_sizes/md.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --fraction 0.2 --name scratch-fvg-2-md --config_file configs/model_sizes/md.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --fraction 0.3 --name scratch-fvg-3-md --config_file configs/model_sizes/md.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --fraction 0.5 --name scratch-fvg-5-md --config_file configs/model_sizes/md.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --fraction 0.7 --name scratch-fvg-7-md --config_file configs/model_sizes/md.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --fraction 1 --name scratch-fvg-10-md --config_file configs/model_sizes/md.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus


# ### SM
# python main.py --fraction 0.1 --name scratch-fvg-1-sm --config_file configs/model_sizes/sm.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --fraction 0.2 --name scratch-fvg-2-sm --config_file configs/model_sizes/sm.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --fraction 0.3 --name scratch-fvg-3-sm --config_file configs/model_sizes/sm.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --fraction 0.5 --name scratch-fvg-5-sm --config_file configs/model_sizes/sm.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --fraction 0.7 --name scratch-fvg-7-sm --config_file configs/model_sizes/sm.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
# python main.py --fraction 1 --name scratch-fvg-10-sm --config_file configs/model_sizes/sm.yaml --trainer wildgait --dataset fvg --model gaitformer --mode run --group from_scratch  --epochs 400 --env exodus
