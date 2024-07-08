# finetune_test


## MODEL yolov7
- doamin : objection detection (object = car & plate)
- get model from : https://github.com/WongKinYiu/yolov7
- get dataset from : roboflow
  - data shape : 640 x 640
  - data format : YOLO v7 PyTorch
  - data file name : cars_w_license_yolov7.tgz

```shell
# prepare python env
cd model_zoo/yolov7
python -m venv venv && source venv/bin/activate
pip install -U pip && pip install -r requirements.txt

# prepare dataset
tar xzf cars_w_license_yolov7.tgz # ./cars_w_license_yolov7

# train
MODEL_NM="yolov7"
DATA_NM="cars_w_license_yolov7"
DATA_PATH="${PWD}/cars_w_license_yolov7/data.yaml"
NOW_TMS=$(date +"%Y%m%d%H%M")
LOG_DIR_PATH="${MODEL_NM}_${DATA_NM}_${NOW_TMS}"
nohup python train.py \
  --name ${LOG_DIR_PATH} \
  --img 640 640 \
  --batch 256 \
  --epochs 100 \
  --data ${DATA_PATH} \
  --weights weights/yolov7.pt \
  --cfg cfg/training/yolov7.yaml \
  --cache --hyp data/hyp.scratch.p5.yaml > logs/${LOG_DIR_PATH}.out 2>&1 &

watch -d nvidia-smi
```


## MODEL llm (Using LLamaFactory)
- doamin : llm
- get model from : ANY of llama factory (huggingface)
- get dataset from : ANY of llama factory (huggingface)

```shell
# prepare python env
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
python -m venv llamafactory && source llamafactory/bin/activate
pip install -U pip
pip install -e ".[torch,metrics]"

# excute gui
llamafactory-cli webui
# http://0.0.0.0:7860/

watch -d nvidia-smi
```