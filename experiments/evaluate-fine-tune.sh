#/bin/bash
set -e
cd ..

##### CASIA Recognition
python evaluate.py  --group fine-tune --name cont-casia-xl-1 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name cont-casia-xl-2 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name cont-casia-xl-3 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name cont-casia-xl-5 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name cont-casia-xl-7 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name cont-casia-xl-10 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog

python evaluate.py  --group fine-tune --name mt-all-casia-sm-1 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-all-casia-sm-2 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-all-casia-sm-3 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-all-casia-sm-5 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-all-casia-sm-7 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-all-casia-sm-10 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog

python evaluate.py  --group fine-tune --name mt-cl-casia-sm-1 --config_file configs/experiments/appearance-clothing-sm.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-cl-casia-sm-2 --config_file configs/experiments/appearance-clothing-sm.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-cl-casia-sm-3 --config_file configs/experiments/appearance-clothing-sm.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-cl-casia-sm-5 --config_file configs/experiments/appearance-clothing-sm.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-cl-casia-sm-7 --config_file configs/experiments/appearance-clothing-sm.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-cl-casia-sm-10 --config_file configs/experiments/appearance-clothing-sm.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog

python evaluate.py  --group fine-tune --name mt-dem-casia-md-1 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-dem-casia-md-2 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-dem-casia-md-3 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-dem-casia-md-5 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-dem-casia-md-7 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-dem-casia-md-10 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/casia-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog

###### FVG Recognition

python evaluate.py  --group fine-tune --name cont-fvg-xl-1 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name cont-fvg-xl-2 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name cont-fvg-xl-3 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name cont-fvg-xl-5 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name cont-fvg-xl-7 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name cont-fvg-xl-10 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog

python evaluate.py  --group fine-tune --name mt-all-fvg-sm-1 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-all-fvg-sm-2 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-all-fvg-sm-3 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-all-fvg-sm-5 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-all-fvg-sm-7 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-all-fvg-sm-10 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog

python evaluate.py  --group fine-tune --name mt-cl-fvg-sm-1 --config_file configs/experiments/appearance-clothing-sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-cl-fvg-sm-2 --config_file configs/experiments/appearance-clothing-sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-cl-fvg-sm-3 --config_file configs/experiments/appearance-clothing-sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-cl-fvg-sm-5 --config_file configs/experiments/appearance-clothing-sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-cl-fvg-sm-7 --config_file configs/experiments/appearance-clothing-sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-cl-fvg-sm-10 --config_file configs/experiments/appearance-clothing-sm.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog

python evaluate.py  --group fine-tune --name mt-dem-fvg-md-1 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-dem-fvg-md-2 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-dem-fvg-md-3 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-dem-fvg-md-5 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-dem-fvg-md-7 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog
python evaluate.py  --group fine-tune --name mt-dem-fvg-md-10 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/fvg-recognition.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-recog

######### CASIA Gender
python evaluate.py  --group fine-tune --name app-all-casia-gender-xl-1 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name app-all-casia-gender-xl-2 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name app-all-casia-gender-xl-3 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name app-all-casia-gender-xl-5 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name app-all-casia-gender-xl-7 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name app-all-casia-gender-xl-10 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender

python evaluate.py  --group fine-tune --name app-dem-casia-gender-xl-1 --config_file configs/experiments/appearance-demographic-xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name app-dem-casia-gender-xl-2 --config_file configs/experiments/appearance-demographic-xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name app-dem-casia-gender-xl-3 --config_file configs/experiments/appearance-demographic-xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name app-dem-casia-gender-xl-5 --config_file configs/experiments/appearance-demographic-xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name app-dem-casia-gender-xl-7 --config_file configs/experiments/appearance-demographic-xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name app-dem-casia-gender-xl-10 --config_file configs/experiments/appearance-demographic-xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender

python evaluate.py  --group fine-tune --name mt-all-casia-gender-sm-1 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name mt-all-casia-gender-sm-2 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name mt-all-casia-gender-sm-3 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name mt-all-casia-gender-sm-5 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name mt-all-casia-gender-sm-7 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name mt-all-casia-gender-sm-10 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender

python evaluate.py  --group fine-tune --name mt-dem-casia-gender-md-1 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name mt-dem-casia-gender-md-2 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name mt-dem-casia-gender-md-3 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name mt-dem-casia-gender-md-5 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name mt-dem-casia-gender-md-7 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender
python evaluate.py  --group fine-tune --name mt-dem-casia-gender-md-10 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune-gender


######## FVG Gender
python evaluate.py  --group fine-tune --name app-all-fvg-gender-xl-1 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name app-all-fvg-gender-xl-2 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name app-all-fvg-gender-xl-3 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name app-all-fvg-gender-xl-5 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name app-all-fvg-gender-xl-7 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name app-all-fvg-gender-xl-10 --config_file configs/model_sizes/xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune

python evaluate.py  --group fine-tune --name app-dem-fvg-gender-xl-1 --config_file configs/experiments/appearance-demographic-xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name app-dem-fvg-gender-xl-2 --config_file configs/experiments/appearance-demographic-xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name app-dem-fvg-gender-xl-3 --config_file configs/experiments/appearance-demographic-xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name app-dem-fvg-gender-xl-5 --config_file configs/experiments/appearance-demographic-xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name app-dem-fvg-gender-xl-7 --config_file configs/experiments/appearance-demographic-xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name app-dem-fvg-gender-xl-10 --config_file configs/experiments/appearance-demographic-xl.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune

python evaluate.py  --group fine-tune --name mt-all-fvg-gender-sm-1 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name mt-all-fvg-gender-sm-2 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name mt-all-fvg-gender-sm-3 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name mt-all-fvg-gender-sm-5 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name mt-all-fvg-gender-sm-7 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name mt-all-fvg-gender-sm-10 --config_file configs/model_sizes/sm.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune

python evaluate.py  --group fine-tune --name mt-dem-fvg-gender-md-1 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name mt-dem-fvg-gender-md-2 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name mt-dem-fvg-gender-md-3 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name mt-dem-fvg-gender-md-5 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name mt-dem-fvg-gender-md-7 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
python evaluate.py  --group fine-tune --name mt-dem-fvg-gender-md-10 --config_file configs/experiments/appearance-demographic-md.yaml --eval_config configs/evaluation/just-gender.yaml --model gaitformer --mode run --env exodus --output_dir fine-tune
